"""
Chat Service — RAG Response-Generation Pipeline
Owner: Member 2 (Farhan)

Pipeline
--------
  user_message
      └─ QueryProcessor.prepare()       — clean + validate input
            └─ retrieval_service.hybrid_search()   — BM25 + HNSW
                  └─ ContextBuilder.build()        — format retrieved chunks
                        └─ LLMClient.generate()    — local / open-weights LLM
                              └─ ChatResponse      — structured response

Multi-turn conversation state is stored in ConversationHistory and injected
into every prompt so the model stays in context across turns.

LLM Backend (no OpenAI)
-----------------------
The service is designed around a pluggable LLMClient interface.
Two concrete implementations are provided:

  1. HuggingFaceLLMClient  — uses a local HF model (e.g. TinyLlama, Mistral)
                             via the `transformers` pipeline.  Good for fully
                             offline / on-prem deployments.

  2. OllamaLLMClient       — delegates to an Ollama server (localhost:11434 by
                             default).  Fastest to set up if you already run
                             Ollama locally or in Docker.

Set LLM_BACKEND in .env to "ollama" (default) or "huggingface".
Set LLM_MODEL   in .env to the model name (e.g. "mistral", "tinyllama").

Integration points
------------------
  - app.services.retrieval_service.hybrid_search   (this module)
  - app.core.logging.get_logger                    (Member 1)
  - Called by app.api.search_chat_api endpoints    (Member 2)
"""

import os
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger
from app.services.retrieval_service import hybrid_search, RetrievedChunk

logger = get_logger(__name__)

# ── Env-driven config ─────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "mistral")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))


# ── DTOs ──────────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    answer: str
    sources: list[RetrievedChunk]
    conversation_id: Optional[str] = None


# ── Conversation history (in-memory; keyed by session/conversation_id) ────────

class ConversationHistory:
    """
    Thread-safe in-memory store for multi-turn chat history.
    Each conversation is a list of ChatMessage objects capped at
    MAX_HISTORY_TURNS * 2 entries (user + assistant per turn).
    """

    def __init__(self):
        self._store: dict[str, list[ChatMessage]] = {}

    def get(self, conversation_id: str) -> list[ChatMessage]:
        return self._store.get(conversation_id, [])

    def append(self, conversation_id: str, message: ChatMessage) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(message)
        # Keep only the most recent N turns
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self._store[conversation_id]) > max_msgs:
            self._store[conversation_id] = self._store[conversation_id][-max_msgs:]

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)


# Singleton instance shared across requests
_history = ConversationHistory()


# ── Query processor ───────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Validates and normalises the user query before retrieval.
    Raises ValueError for clearly invalid inputs.
    """

    MAX_QUERY_LEN = 1000

    @classmethod
    def prepare(cls, raw_query: str) -> str:
        if not raw_query or not isinstance(raw_query, str):
            raise ValueError("Query must be a non-empty string.")
        q = raw_query.strip()
        if len(q) == 0:
            raise ValueError("Query must not be blank.")
        if len(q) > cls.MAX_QUERY_LEN:
            raise ValueError(
                f"Query too long ({len(q)} chars). Max allowed: {cls.MAX_QUERY_LEN}."
            )
        return q


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Converts a list of RetrievedChunk objects into a prompt-ready context
    block, respecting the MAX_CONTEXT_CHARS budget.
    """

    @staticmethod
    def build(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        if not chunks:
            return "No relevant documents were found."

        parts: list[str] = []
        budget = max_chars

        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {i}] {chunk.file_name} — page {chunk.page_number}"
            )
            body = chunk.content.strip()
            entry = f"{header}\n{body}\n"

            if len(entry) > budget:
                # Truncate the body to fit within budget
                available = budget - len(header) - 4  # '\n' chars
                if available <= 0:
                    break
                entry = f"{header}\n{body[:available]}…\n"

            parts.append(entry)
            budget -= len(entry)

            if budget <= 0:
                break

        return "\n---\n".join(parts)


# ── LLM client abstraction ────────────────────────────────────────────────────

class LLMClient:
    """Base interface for LLM backends."""

    def generate(self, system_prompt: str, history: list[ChatMessage], user_message: str) -> str:
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    """
    Calls a locally running Ollama server via its HTTP API.
    Install Ollama: https://ollama.com
    Then: ollama pull mistral
    """

    def __init__(self, model: str = LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, history: list[ChatMessage], user_message: str) -> str:
        import requests  # stdlib-only dependency — already in requirements.txt

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"OllamaLLMClient.generate failed: {exc}")
            return (
                "I'm sorry, I couldn't generate a response right now. "
                "Please ensure the Ollama service is running and try again."
            )


class HuggingFaceLLMClient(LLMClient):
    """
    Runs a local HuggingFace model via the `transformers` library.
    Suitable for fully offline / air-gapped deployments.

    Set LLM_MODEL to any causal-LM on HuggingFace Hub, e.g.:
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        "mistralai/Mistral-7B-Instruct-v0.2"  (requires ~14 GB RAM)
    """

    def __init__(self, model_name: str = LLM_MODEL):
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
            )
            logger.info(f"HuggingFaceLLMClient loaded model: {model_name}")
        except Exception as exc:
            logger.error(f"HuggingFaceLLMClient init failed: {exc}")
            raise

    def generate(self, system_prompt: str, history: list[ChatMessage], user_message: str) -> str:
        # Build a simple [INST] prompt (works for most instruction-tuned models)
        turns = ""
        for msg in history:
            if msg.role == "user":
                turns += f"[INST] {msg.content} [/INST]\n"
            else:
                turns += f"{msg.content}\n"

        prompt = (
            f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{turns}"
            f"[INST] {user_message} [/INST]\n"
        )

        try:
            outputs = self._pipe(prompt)
            generated = outputs[0]["generated_text"]
            # Strip the prompt prefix that some models echo back
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            return generated.strip()
        except Exception as exc:
            logger.error(f"HuggingFaceLLMClient.generate failed: {exc}")
            return "I couldn't generate a response. Please check the LLM configuration."


def _build_llm_client() -> LLMClient:
    if LLM_BACKEND == "huggingface":
        return HuggingFaceLLMClient(model_name=LLM_MODEL)
    # Default: Ollama
    return OllamaLLMClient(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


# Lazy singleton — instantiated on first call to avoid startup cost when unused
_llm_client: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = _build_llm_client()
    return _llm_client


# ── RAG system prompt ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a precise and intelligent document assistant.
    Your job is to answer the user's question using ONLY the provided context.

    ====================
    STRICT RULES
    ====================
    1. Do NOT copy or dump the context or repeat full paragraphs.
    2. Use ONLY the definitions provided in the clauses. Do not use outside knowledge.
    3. Summarize and synthesize information into a clean, final answer.
    4. Only use the MOST relevant parts of the context; ignore irrelevant text.
    5. Do NOT hallucinate. If the answer is not in the context, say: "The provided documents do not contain sufficient information."
    6. CITATIONS: Always cite sources inline exactly as they appear in the context headers, e.g., ([Source 1]).
    7. NO CHATTER: Do not mention "Source X says" or "Based on the documents" in your prose.
    8. FORMATTING: Return plain text with standard line breaks. Do not return raw JSON escape characters like \\n or \\".
    9. Use ONLY the MOST relevant parts of the context; ignore unrelated chunks completely.
    10. - If multiple chunks conflict, prefer the most specific one.                                      

    ====================
    ANSWER FORMAT
    ====================
    - Start directly with the answer (no intro or "Okay, here is the answer").
    - Be concise and professional.
    - Use bullet points ONLY if helpful for lists.

    ====================
    DOCUMENT CONTEXT
    ====================
    {context}
    ====================
    END OF CONTEXT
    ====================
""")

# ── Public chat API ───────────────────────────────────────────────────────────

def chat(
    user_message: str,
    conversation_id: str = "default",
    top_k: int = 3,
    bm25_weight: float = 0.2,
    hnsw_weight: float = 0.8,
) -> ChatResponse:
    """
    Full RAG pipeline:
      1. Validate + clean the user query.
      2. Hybrid search (BM25 + HNSW) to retrieve relevant chunks.
      3. Build a context block from the chunks.
      4. Inject context + conversation history into LLM prompt.
      5. Generate and return a structured ChatResponse.

    Parameters
    ----------
    user_message     : The user's question.
    conversation_id  : Session identifier for multi-turn history tracking.
    top_k            : Number of chunks to retrieve (default 5).
    bm25_weight      : BM25 fusion weight (default 0.4).
    hnsw_weight      : HNSW fusion weight (default 0.6).

    Returns
    -------
    ChatResponse  with .answer (str) and .sources (list[RetrievedChunk]).
    """
    # ── 1. Validate ───────────────────────────────────────────────────────────
    query = QueryProcessor.prepare(user_message)

    # ── 2. Retrieve ───────────────────────────────────────────────────────────
    chunks = hybrid_search(query, top_k=top_k, bm25_weight=bm25_weight, hnsw_weight=hnsw_weight)

    # ── 3. Build context ──────────────────────────────────────────────────────
    context = ContextBuilder.build(chunks)

    # ── 4. Build system prompt ────────────────────────────────────────────────
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # ── 5. Retrieve conversation history ──────────────────────────────────────
    history = _history.get(conversation_id)

    # ── 6. Generate answer ────────────────────────────────────────────────────
    llm = _get_llm()
    answer = llm.generate(system_prompt, history, query)

    # ── 7. Persist turn to history ────────────────────────────────────────────
    _history.append(conversation_id, ChatMessage(role="user", content=query))
    _history.append(conversation_id, ChatMessage(role="assistant", content=answer))

    logger.info(
        f"chat: conv={conversation_id} | query='{query[:60]}' | "
        f"sources={len(chunks)} | answer_len={len(answer)}"
    )

    return ChatResponse(
        answer=answer,
        sources=chunks,
        conversation_id=conversation_id,
    )


def clear_history(conversation_id: str) -> None:
    """Remove all stored turns for *conversation_id*."""
    _history.clear(conversation_id)
    logger.info(f"Cleared conversation history for: {conversation_id}")

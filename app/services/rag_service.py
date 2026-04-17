# from app.services.embedding_service import generate_embedding
# from app.services.db_service import hybrid_search_db
# from app.services.llm_service import generate_answer

# def to_pgvector(vec):
#     return "[" + ",".join(map(str, vec)) + "]"

# def hybrid_search(query: str, top_k: int = 5):
#     query_embedding = generate_embedding(query)
#     query_embedding_str = to_pgvector(query_embedding)

#     return hybrid_search_db(query, query_embedding_str, top_k)



# def build_prompt(query: str, contexts: list[dict]) -> str:
#     context_text = "\n\n".join(
#         [f"(Page {c['page_number']}) {c['content']}" for c in contexts]
#     )

#     return f"""
# You are a strict document question answering system.

# Rules:
# - Answer ONLY using the context below
# - Do NOT add external knowledge
# - Do NOT generalize or assume
# - Every statement MUST include (Page X)
# - If answer not found, say: "Not found in document"

# Context:
# {context_text}

# Question:
# {query}
# """

# def rag_query(query: str):
#     results = hybrid_search(query, top_k=5)

#     prompt = build_prompt(query, results)

#     answer = generate_answer(prompt)

#     return {
#         "answer": answer,
#         "sources": results
#     }
"""
Retrieval Service — Hybrid BM25 (sparse) + HNSW (dense) search
Owner: Member 2 (Farhan)

Architecture
------------
  query
    ├─ BM25  → PostgreSQL full-text search (tsvector / GIN index)
    └─ HNSW  → pgvector HNSW cosine-similarity index
          └─ Reciprocal Rank Fusion (RRF) → unified ranked list
                └─ top-k RetrievedChunk list

No external search engines required — everything runs inside the existing
PostgreSQL + pgvector container already in docker-compose.yml.

Integration points
------------------
  - Uses app.db.db.get_connection()          (Member 1)
  - Uses app.services.embedding_service      (Member 2 — this module's sibling)
  - Called by app.services.chat_service      (Member 2)
  - Called by app.api.search_chat_api        (Member 2)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.db.db import get_connection
from app.services.embedding_service import generate_embedding

logger = logging.getLogger(__name__)

# ── Tuning knobs (can be overridden per call) ─────────────────────────────────
DEFAULT_TOP_K: int = 5
DEFAULT_BM25_WEIGHT: float = 0.4   # α  — weight for BM25 rank contribution
DEFAULT_HNSW_WEIGHT: float = 0.6   # β  — weight for HNSW rank contribution
RRF_K: int = 60                     # RRF constant (60 is the standard default)


# ── Data transfer object ──────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single ranked result returned by hybrid search."""
    id: str
    document_id: str
    file_name: str
    page_number: int
    chunk_id: int
    content: str
    bm25_rank: Optional[int] = None    # 1-based rank from BM25 leg
    hnsw_rank: Optional[int] = None    # 1-based rank from HNSW leg
    rrf_score: float = 0.0             # final fusion score (higher = better)
    metadata: dict = field(default_factory=dict)


# ── BM25 retrieval (PostgreSQL full-text search) ──────────────────────────────

def _bm25_search(query: str, top_k: int = DEFAULT_TOP_K * 2) -> list[RetrievedChunk]:
    """
    Use PostgreSQL's tsvector + tsquery full-text search (GIN-indexed).
    Returns up to *top_k* rows ordered by ts_rank_cd descending.
    """
    if not query or not query.strip():
        return []

    sql = """
        SELECT
            id,
            document_id::text,
            file_name,
            page_number,
            chunk_id,
            content,
            ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS bm25_score
        FROM embeddings
        ORDER BY bm25_score DESC
        LIMIT %s;
    """

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql, (query, top_k))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.error(f"BM25 search failed: {exc}")
        return []

    results = []
    for row in rows:
        chunk = RetrievedChunk(
            id=row[0],
            document_id=row[1],
            file_name=row[2] or "",
            page_number=row[3] or 0,
            chunk_id=row[4] or 0,
            content=row[5] or "",
        )
        results.append(chunk)

    logger.debug(f"BM25 returned {len(results)} results for query: {query[:60]}")
    return results


# ── HNSW retrieval (pgvector cosine similarity) ───────────────────────────────

def _hnsw_search(query: str, top_k: int = DEFAULT_TOP_K * 2) -> list[RetrievedChunk]:
    """
    Encode the query with all-MiniLM-L6-v2 and run HNSW ANN search via
    pgvector's <=> cosine-distance operator (HNSW index on embeddings table).
    Returns up to *top_k* rows ordered by cosine similarity descending.
    """
    if not query or not query.strip():
        return []

    embedding = generate_embedding(query)
    if embedding is None:
        logger.warning("HNSW search skipped: could not generate query embedding")
        return []

    # pgvector expects a string like '[0.1, 0.2, ...]'
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    sql = """
        SELECT
            id,
            document_id::text,
            file_name,
            page_number,
            chunk_id,
            content,
            1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM embeddings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql, (vec_str, vec_str, top_k))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.error(f"HNSW search failed: {exc}")
        return []

    results = []
    for row in rows:
        chunk = RetrievedChunk(
            id=row[0],
            document_id=row[1],
            file_name=row[2] or "",
            page_number=row[3] or 0,
            chunk_id=row[4] or 0,
            content=row[5] or "",
        )
        results.append(chunk)

    logger.debug(f"HNSW returned {len(results)} results for query: {query[:60]}")
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    bm25_results: list[RetrievedChunk],
    hnsw_results: list[RetrievedChunk],
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    hnsw_weight: float = DEFAULT_HNSW_WEIGHT,
    rrf_k: int = RRF_K,
) -> list[RetrievedChunk]:
    """
    Combine two ranked lists using weighted Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = α * Σ 1/(k + rank_bm25(d))  +  β * Σ 1/(k + rank_hnsw(d))

    Documents not appearing in one list are simply scored 0 for that leg.
    """
    # Build lookup: chunk_id → RetrievedChunk (HNSW result preferred for content)
    merged: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(bm25_results, start=1):
        chunk.bm25_rank = rank
        chunk.rrf_score += bm25_weight / (rrf_k + rank)
        merged[chunk.id] = chunk

    for rank, chunk in enumerate(hnsw_results, start=1):
        chunk.hnsw_rank = rank
        rrf_contribution = hnsw_weight / (rrf_k + rank)
        if chunk.id in merged:
            merged[chunk.id].hnsw_rank = rank
            merged[chunk.id].rrf_score += rrf_contribution
        else:
            chunk.rrf_score += rrf_contribution
            merged[chunk.id] = chunk

    fused = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    return fused


# ── Public hybrid search entry point ─────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    hnsw_weight: float = DEFAULT_HNSW_WEIGHT,
) -> list[RetrievedChunk]:
    """
    Run BM25 + HNSW in parallel, fuse via RRF, return the top-*top_k* chunks.

    Parameters
    ----------
    query       : User's natural-language question.
    top_k       : Number of final chunks to return after fusion.
    bm25_weight : α — contribution weight for the BM25 leg (default 0.4).
    hnsw_weight : β — contribution weight for the HNSW leg (default 0.6).

    Returns
    -------
    list[RetrievedChunk]
        Sorted by descending RRF score; at most *top_k* entries.
    """
    if not query or not query.strip():
        logger.warning("hybrid_search called with empty query")
        return []

    # Fetch a wider pool from each leg so fusion has enough candidates
    pool = top_k * 3

    bm25_results = _bm25_search(query, top_k=pool)
    hnsw_results = _hnsw_search(query, top_k=pool)

    fused = _reciprocal_rank_fusion(
        bm25_results,
        hnsw_results,
        bm25_weight=bm25_weight,
        hnsw_weight=hnsw_weight,
    )

    final = fused[:top_k]
    logger.info(
        f"hybrid_search: query='{query[:60]}' | "
        f"bm25={len(bm25_results)} hnsw={len(hnsw_results)} fused={len(fused)} returned={len(final)}"
    )
    return final
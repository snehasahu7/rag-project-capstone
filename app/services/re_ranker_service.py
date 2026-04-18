from sentence_transformers import CrossEncoder

# load once (important)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list):
    if not chunks:
        return chunks

    pairs = [(query, c.content) for c in chunks]
    scores = reranker_model.predict(pairs)

    # assign scores (use separate field ideally)
    for c, score in zip(chunks, scores):
        c.rerank_score = float(score)

    # 🔥 filter only positive scores
    filtered = [c for c in chunks if c.rerank_score > 0]

    # fallback if everything removed
    if not filtered:
        filtered = chunks

    return sorted(filtered, key=lambda x: x.rerank_score, reverse=True)
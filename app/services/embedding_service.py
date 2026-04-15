from sentence_transformers import SentenceTransformer

# load once (important)
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str):
    if not text or not text.strip():
        return None
    return model.encode(text).tolist()
# utils/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None

def load_model(name="all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL

def embed_text(text):
    model = load_model()
    if isinstance(text, str):
        text = [text]
    emb = model.encode(text, convert_to_numpy=True)
    if emb.ndim == 1:
        return emb
    return emb[0]

def cosine_sim(a, b):
    a = a.astype(float)
    b = b.astype(float)
    num = float(np.dot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    sim = num/den
    sim = max(0.0, min(1.0, sim))
    return sim

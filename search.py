import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME = "all-MiniLM-L6-v2"


embed_model = SentenceTransformer(MODEL_NAME)


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/meta.json"
CHUNK_PATH = "faiss_store/chunks.json"

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

with open(CHUNK_PATH, "r") as f:
    all_chunks = json.load(f)


def embed(text):
    """Embed text using SentenceTransformer."""
    return embed_model.encode([text], convert_to_numpy=True).astype("float32")


def semantic_search(query, k=5):
    """FAISS search + reranking."""
    q_vec = embed(query)

    
    D, I = index.search(q_vec, k)

    candidates = []
    for dist, idx in zip(D[0], I[0]):
        candidates.append({
            "distance": float(dist),
            "source": metadata[idx]["source"],
            "chunk_index": metadata[idx]["chunk_index"],
            "text": all_chunks[idx]
        })

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    # Attach scores and sort by relevance
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:k]


# Debug only
if __name__ == "__main__":
    q = input("Query: ")
    print(semantic_search(q))

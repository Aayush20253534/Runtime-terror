import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ===========================================================
# 1. YOUR SINGLE SOURCE STRING (replace this content)
# ===========================================================
SOURCE_TEXT = """
Artificial intelligence has rapidly evolved, enabling advancements in deep learning, 
autonomous robotics, natural language processing, and healthcare diagnostics. 
Modern AI models such as transformers, GPT, and multimodal systems demonstrate 
the ability to understand language, images, and structured data with high accuracy. 
Reinforcement learning drives improvements in robotic control and autonomous navigation, 
while retrieval-augmented generation (RAG) helps reduce hallucinations in LLMs by grounding 
answers in external knowledge sources.
"""


# ===========================================================
# 2. CHUNKING FUNCTION
# ===========================================================
def chunk_text(text, size=300, overlap=50):
    """Smart sentence-based chunker."""
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()

        if len(" ".join(current + words)) > size:
            chunks.append(" ".join(current))
            current = words[-overlap:]
        else:
            current.extend(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


# ===========================================================
# 3. CHUNK THE SINGLE STRING
# ===========================================================
chunks = chunk_text(SOURCE_TEXT)
# print(f"Total chunks created: {len(chunks)}")


# ===========================================================
# 4. EMBEDDING + BUILDING FAISS
# ===========================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("FAISS index built.")


# ===========================================================
# 5. RERANKER (Cross Encoder)
# ===========================================================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ===========================================================
# 6. SEMANTIC SEARCH + RERANKING
# ===========================================================
def semantic_search(query, k=3):
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, k)

    candidates = []
    for dist, idx in zip(D[0], I[0]):
        candidates.append({
            "text": chunks[idx],
            "distance": float(dist)
        })

    # Rerank based on relevance
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["score"] = float(s)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[:k]


# ===========================================================
# 7. TEST QUERY
# ===========================================================
query = input("Ask something: ")

results = semantic_search(query)

print("\nTop Relevant Chunks:\n")
for i, r in enumerate(results, 1):
    # print(f"Rank {i}")
    # print("Score:", r["score"])
    print("Text:", r["text"])
    print()

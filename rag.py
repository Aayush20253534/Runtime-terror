import faiss
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

class RAGSystem:
    def __init__(self):
        print("Loading embedding models...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.index = None
        self.chunks = []
        print("Models loaded.")
 
    def chunk_text(self, text, size=300, overlap=50):
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

    def ingest(self, text):
        self.chunks = self.chunk_text(text)
        if not self.chunks:
            print("Warning: No text chunks created.")
            return

        embeddings = self.embed_model.encode(self.chunks, convert_to_numpy=True).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"Index built with {len(self.chunks)} chunks.")

    def semantic_search(self, query, k=3):
        if self.index is None or not self.chunks:
            return []
        q_vec = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_vec, k)
        candidates = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.chunks):
                candidates.append({
                    "text": self.chunks[idx],
                    "distance": float(dist)
                })
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def generate_answer(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "I couldn't find any relevant information in the document."
        context = "\n\n".join([c["text"] for c in retrieved_chunks])
        return f"Based on the document, here is the relevant information:\n\n{context}"
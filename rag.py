import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


class RAGSystem:

    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = None
        self.chunks = []

        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise Exception("GEMINI_API_KEY missing.")

        self.client = genai.Client(api_key=self.api_key)

    def chunk_text(self, text, size=800):
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current = ""

        for p in paragraphs:
            if len(current) + len(p) > size:
                chunks.append(current)
                current = p
            else:
                current += " " + p if current else p

        if current:
            chunks.append(current)

        return chunks

    def ingest(self, text):
        self.chunks = self.chunk_text(text)

        if not self.chunks:
            print("No chunks to index.")
            return

        embeddings = self.embed_model.encode(self.chunks)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print(f"[RAG] Indexed {len(self.chunks)} chunks.")

    def semantic_search(self, query, k=5):
        if self.index is None:
            return []

        q_vec = self.embed_model.encode([query]).astype("float32")
        D, I = self.index.search(q_vec, k)

        retrieved = []
        for idx in I[0]:
            if idx < len(self.chunks):
                retrieved.append({"text": self.chunks[idx]})

        return retrieved

    def generate_answer(self, query, retrieved):
        if not retrieved:
            return "No relevant information found."

        context = "\n\n".join([c["text"] for c in retrieved])

        prompt = f"""
Use ONLY this context to answer the question.

CONTEXT:
{context}

QUESTION:
{query}
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

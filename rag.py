"""Python file that implements RAG logic and sematic search."""

import faiss
import nltk
import os
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv
from pathlib import Path

# Make sure nltk has the sentence tokenizer (not strictly needed now, but ok)
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

        # Load Gemini API key
        env_path = Path(__file__).parent / "API_key.env"
        load_dotenv(env_path)
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in environment!")
        self.client = genai.Client(api_key=self.api_key)

        print("Models loaded.")

    def chunk_text(self, text, size=800):
        """
        Split text into chunks based on paragraphs.
        """

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        chunks = []
        current = ""

        for p in paragraphs:
            if len(current) + len(p) > size:
                if current.strip():
                    chunks.append(current.strip())
                current = p  # start new chunk
            else:
                if current:
                    current += " " + p
                else:
                    current = p

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def ingest(self, text):
        """
        Store embeddings in a FAISS index for fast search.
        """
        self.chunks = self.chunk_text(text)

        print(f"[INGEST] Created {len(self.chunks)} chunks.")
        if len(self.chunks) > 0:
            print("[INGEST] Example chunk:")
            print(self.chunks[0][:300], "...\n")

        if not self.chunks:
            print("[INGEST] Warning: No text chunks created.")
            self.index = None
            return

        embeddings = self.embed_model.encode(
            self.chunks, convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print(f"[INGEST] FAISS index built with {len(self.chunks)} chunks.")

    def extract_core_term(self, query: str) -> str:
        """
        Try to extract the main term from questions like:
        - "What is another name for mitochondria?"
        - "Other name for ATP?"
        - "What is the alternate name of glycolysis?"

        If we can't find any pattern, we just return the whole query.
        """
        q = query.lower()

        # Common patterns for synonym-style questions
        patterns = [
            r"another name for\s+(.+)\??",
            r"other name for\s+(.+)\??",
            r"alternate name for\s+(.+)\??",
            r"alternative name for\s+(.+)\??",
            r"synonym for\s+(.+)\??",
            r"also called\s+(.+)\??",
        ]

        for pat in patterns:
            m = re.search(pat, q)
            if m:
                term = m.group(1).strip()
                # Remove trailing question marks / punctuation
                term = term.strip(" ?.,")
                return term

        # If no pattern matched, just return original query
        return query.strip(" ?.")

    def expand_query(self, query):
        """
        Expand user query into multiple forms to improve recall.

        for example:
            Query: "What is another name for X?"
            -> Core term extracted: "X"
            -> Expanded queries include:
                "X"
                "X also called"
                "X also known as"
                "alternate name for X"
                etc.
        """
        core_term = self.extract_core_term(query)

        expansions = [
            query,  # full question as-is
            core_term,  # just the main term, e.g. "mitochondria"
            f"{core_term} also called",
            f"{core_term} also known as",
            f"{core_term} referred to as",
            f"{core_term} also named",
            f"{core_term} is called",
            f"alternate name for {core_term}",
            f"synonym of {core_term}",
            f"{core_term} aka",
            f"{core_term} is also called",
        ]

        # Remove duplicates & empty strings
        expansions = [e.strip() for e in expansions if e.strip()]
        unique = list(dict.fromkeys(expansions))  # keep order, remove dups

        print(f"[QUERY] Original query: {query}")
        print(f"[QUERY] Core term: {core_term}")
        print(f"[QUERY] Expanded queries:")
        for e in unique:
            print("   -", e)
        print()

        return unique

    def semantic_search(self, query, k=20):
        """
        1. Expand the query (to better handle synonyms).
        2. For each expanded query, retrieve top-k chunks from FAISS.
        3. Collect all candidate chunks.
        4. Rerank candidates using CrossEncoder (query, chunk) pairs.
        5. Return the top 5 chunks.
        """
        if self.index is None or not self.chunks:
            print("[SEARCH] No index or chunks available. Did you call ingest()?")
            return []

        expanded_queries = self.expand_query(query)
        candidates = []

        for eq in expanded_queries:
            q_vec = self.embed_model.encode(
                [eq], convert_to_numpy=True
            ).astype("float32")

            D, I = self.index.search(q_vec, k)

            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(self.chunks):
                    candidates.append(
                        {
                            "text": self.chunks[idx],
                            "distance": float(dist),
                        }
                    )

        if not candidates:
            print("[SEARCH] No candidates retrieved from FAISS.")
            return []

        print(f"[SEARCH] Retrieved {len(candidates)} raw candidates before reranking.")

        # Rerank with CrossEncoder
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["score"] = float(s)

        # Sort candidates by reranker score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Optionally, we can deduplicate by text
        seen = set()
        deduped = []
        for c in candidates:
            t = c["text"]
            if t not in seen:
                seen.add(t)
                deduped.append(c)

        top_k = deduped[:5]
        print(f"[SEARCH] Top {len(top_k)} candidates after reranking + dedupe.\n")

        return top_k

    def generate_answer(self, query, retrieved_chunks):
        """
        Use Gemini to synthesize an answer from the retrieved chunks.

        If no chunks are retrieved, return a friendly fallback.
        """
        if not retrieved_chunks:
            return "I couldn't find any relevant information in the document."

        context = "\n\n".join(c["text"] for c in retrieved_chunks)

        prompt = f"""
You are an assistant. Use ONLY the following CONTEXT (from a document) to answer the QUESTION.

- If the answer is present in the context, answer clearly.
- If the document gives another name / synonym, mention it explicitly.
- If the answer is NOT in the context, say: "The document does not specify this."

CONTEXT:
{context}

QUESTION:
{query}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print("[LLM] Error while calling Gemini:", e)
            return "There was an error generating the answer from the language model."

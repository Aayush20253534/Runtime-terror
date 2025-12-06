import os
import glob
import json
import faiss
from sentence_transformers import SentenceTransformer
DATA_DIR = "data/docs"
INDEX_DIR = "faiss_store"
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = 300
OVERLAP = 50
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
all_chunks = []
metadata = []

def chunk_text(text, size=300, overlap=50):
    """Smarter chunking: splits by sentences but respects chunk size."""
    import nltk

    # Download both required tokenizers
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()

        if len(" ".join(current + words)) > size:
            chunks.append(" ".join(current))
            current = words[-overlap:]  # overlap
        else:
            current.extend(words)

    if current:
        chunks.append(" ".join(current))

    return chunks
files = glob.glob(f"{DATA_DIR}/*.txt")
if not files:
    raise RuntimeError("No .txt files found in data/docs/")

print("Found files:", files)

# Load and chunk every file
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read().strip()

    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

    for idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata.append({
            "source": os.path.basename(fpath),
            "chunk_index": idx
        })

print("Total chunks:", len(all_chunks))

# EMBEDDING
print("\nEmbedding...")
embeddings = model.encode(
    all_chunks,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index + metadata
faiss.write_index(index, f"{INDEX_DIR}/index.faiss")

with open(f"{INDEX_DIR}/meta.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Save chunks themselves for retrieval
with open(f"{INDEX_DIR}/chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

print("\n✅ Completed building FAISS index!")

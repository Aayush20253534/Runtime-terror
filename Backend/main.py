from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import json
from Extracter import TextExtracter
from rag import RAGSystem
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LAST_RESULT = {}
rag_system = RAGSystem()

def process_file_backend(file_path: str):
    extractor = TextExtracter()
    result = extractor.handleFiles(file_path) 
    return result

def get_database_data():
    db_path = "Database/data.json"
    if not os.path.exists(db_path):
        return {}
    try:
        with open(db_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}
    
@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    global LAST_RESULT
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        processed = process_file_backend(file_path)

        summary = processed.get("summary", "")
        full_text = processed.get("text", summary) 
        category = processed.get("category", "Unknown")
        date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        LAST_RESULT = {
            "summary": summary,
            "category": category,
            "date": date_now
        }

        if full_text:
            print("Ingesting text into RAG...")
            rag_system.ingest(full_text)
        else:
            print("Warning: No text found to ingest.")

        return {
            "status": "success",
            "message": "File processed and indexed successfully",
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
async def get_summary():
    if not LAST_RESULT.get("summary"):
        return {"summary": "No summary yet.", "category": "-", "date": "-"}
    return LAST_RESULT

@app.get("/ask")
async def ask(query: str):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        retrieved = rag_system.semantic_search(query)
        answer = rag_system.generate_answer(query, retrieved)
        retrieved_texts = [c["text"] for c in retrieved]
        return {
            "query": query,
            "retrieved_chunks": retrieved_texts,
            "answer": answer
        }
    except Exception as e:
        print(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    data = get_database_data()
    return data

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    data = get_database_data()
    if doc_id not in data:
        raise HTTPException(status_code=404, detail="Document not found")
    return data[doc_id]

@app.get("/analysis")
async def get_analysis():
    """Calculates statistics for the dashboard."""
    data = get_database_data()
    
    total_docs = len(data)
    
    # Extract all categories
    categories = [item.get("category", "Unknown") for item in data.values()]
    
    # Count frequency of each category
    category_counts = Counter(categories)
    
    # Sort by most frequent
    sorted_categories = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))

    return {
        "total_documents": total_docs,
        "unique_categories": len(category_counts),
        "category_counts": sorted_categories
    }
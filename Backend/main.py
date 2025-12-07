from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
from Extracter import TextExtracter
from rag import RAGSystem

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
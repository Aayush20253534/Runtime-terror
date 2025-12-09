from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import json

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

os.makedirs("uploads", exist_ok=True)
os.makedirs("Database", exist_ok=True)

if not os.path.exists("Database/data.json"):
    with open("Database/data.json", "w") as f:
        json.dump({}, f)

rag_system = RAGSystem()
LAST_RESULT = {}


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        extractor = TextExtracter()
        processed = extractor.handleFiles(file_path)

        summary = processed.get("summary", "")
        category = processed.get("category", "Unknown")

        LAST_RESULT.update({
            "summary": summary,
            "category": category,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        rag_system.ingest(processed.get("text", ""))

        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary")
async def get_summary():
    if not LAST_RESULT:
        return {"summary": "No summary yet.", "category": "-", "date": "-"}
    return LAST_RESULT


@app.get("/ask")
async def ask(query: str):
    retrieved = rag_system.semantic_search(query)
    answer = rag_system.generate_answer(query, retrieved)
    return {
        "query": query,
        "retrieved_chunks": [r["text"] for r in retrieved],
        "answer": answer
    }


@app.get("/history")
async def get_history():
    with open("Database/data.json", "r") as f:
        return json.load(f)


@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    with open("Database/data.json", "r") as f:
        data = json.load(f)

    if doc_id not in data:
        raise HTTPException(status_code=404, detail="Document not found")

    return data[doc_id]

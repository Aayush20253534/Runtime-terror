from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
from Extracter import TextExtracter

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
LAST_RESULT ={}

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
        category = processed.get("category", "Unknown")
        date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LAST_RESULT = {
            "summary": summary,
            "category": category,
            "date": date_now
        }
        return {
            "status": "success",
            "message": "File processed successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary")
async def get_summary():
    if not LAST_RESULT["summary"]:
        raise HTTPException(status_code=404, detail="No summary found. Upload a file first.")

    return LAST_RESULT

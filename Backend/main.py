from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
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

def process_file_backend(file_path: str):
    print("Background processing started:", file_path)

    extractor = TextExtracter()        # create object
    result = extractor.handleFiles(file_path)   # <-- CALL FUNCTION HERE

    print("Background processing finished.")
    print(result)

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        background_tasks.add_task(process_file_backend, file_path)
        return {
            "status": "success",
            "message": "File uploaded successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

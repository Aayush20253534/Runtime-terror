import fitz  # PyMuPDF
import os
import json
import pytesseract
from PIL import Image
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from google import genai


class GenerationModel:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it in Render environment.")

        self.client = genai.Client(api_key=self.api_key)


class TextExtracter:

    def __init__(self):
        self.model = GenerationModel()

    def handleFiles(self, file: str) -> dict:

        if file.endswith(".pdf"):
            self.text = self.extractText(file)
        else:
            self.text = self.image_OCR(file)

        self.summary, self.category = self.summarize_categorize(self.text)

        # Ensure DB folder exists
        os.makedirs("Database", exist_ok=True)

        try:
            with open("Database/data.json", "r") as f:
                data_stored = json.load(f)
        except FileNotFoundError:
            data_stored = {}

        index = f"DOC_{len(data_stored)+1:03d}"

        data_entry = {
            index: {
                "filename": os.path.basename(file),
                "text": self.text,
                "summary": self.summary,
                "date": f"{date.today()}",
                "category": self.category,
                "vector_id": index
            }
        }

        data_stored[index] = data_entry[index]

        with open("Database/data.json", "w") as f:
            json.dump(data_stored, f, indent=4)

        return data_stored[index]

    def extractText(self, filePath: str) -> str:
        text = ""
        with fitz.open(filePath) as doc:
            for page in doc:
                text += page.get_text() + "\n\n"
        return text.strip()

    def image_OCR(self, imagePath: str) -> str:
        img = Image.open(imagePath)
        text = pytesseract.image_to_string(img)
        return text.strip()

    def summarize_categorize(self, text):
        # --- Summary Prompt ---
        prompt_summary = f"""
Summarize the following document. The summary must be plain text.
Also add simple bullet points at the end.

Document:
{text}
"""

        summary = self.model.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_summary
        ).text

        # --- Category Prompt ---
        prompt_category = f"""
Classify the document into EXACTLY one of these:

Operations
Enemy sightings
Zord Systems
Ranger Personnel
Research & Development
Weapons & Equipment
Communications
Administration
Security
Archives
Logistics
Infrastructure & Maintenance

Return ONLY the category word.

Document:
{text}
"""

        category = self.model.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_category
        ).text.strip()

        return summary, category

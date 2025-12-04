'''It deals with the functions related to text extraction from PDFs and OCR for images.'''

import pymupdf
import os
import google.generativeai as genai
import easyocr
import json
 
from datetime import date
from dotenv import load_dotenv

class GenerationModel:

    def __init__(self) -> None:

        load_dotenv("Backend/API_key.env")
        self.api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")  

class TextExtracter:


    def __init__(self) -> None:

        self.model = GenerationModel()
        self.possibleCategories = ["Operations","Enemy sightings","Zord Systems","Ranger Personnel","Research & Development", "Weapons & Equipment", "Communications", "Administration", "Security", "Archives", "Logistics", "Infrastructure & Maintenance"]

    def handleFiles(self, file):
        
        if file.endswith(".pdf"):
            self.text = self.extractText(file)
        
        else:
            self.text = self.image_OCR(file)
        
        self.summary, self.category = self.summarize_categorize(self.text)
        
        

        data_entry = \
        {
            file:
            {
                "filename" : os.path.basename(file),
                "text" : self.text,
                "summary" : self.summary,
                "date" : f"{date.today()}",
                "category" : self.category,
                "vector_id" : "file1"
            }
        }

        try:
            with open("Database/data.json", "r") as f:
                data_stored = json.load(f)
        
        except FileNotFoundError:
            data_stored = {}

        data_stored[file] = data_entry[file]

        with open("Database/data.json", "w") as f:
            json.dump(data_stored, f, indent=4)

        return data_stored
    
    def extractText(self, filePath:str) -> str:
        
        with pymupdf.open(filePath) as doc:

            text = chr(12).join([page.get_text() for page in doc])
        
        return text

    def image_OCR(self, imagePath:str) -> str:
        
        reader = easyocr.Reader(['en'])
        text = reader.readtext(imagePath)
        lines = [entry[1] for entry in text]

        text = "\n".join(lines)

        return text


    def summarize_categorize(self,text) -> str:

        prompt_summary = f"""{text}\nSummarize this text in short while keeping in all the important topics intact. Also provide bullet points at the end."""
        summary = self.model.model.generate_content(prompt_summary)

        prompt_category = f"""{text}\nGo through the above text and assign it a category out of these options - {self.possibleCategories}"""
        category = self.model.model.generate_content(prompt_category)

        return summary.text, category.text
        

if __name__ == "__main__":

    extracter = TextExtracter()


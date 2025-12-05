'''It deals with the functions related to text extraction from PDFs and OCR for images.'''

import pymupdf
import os
from pathlib import Path
import google.generativeai as genai
import easyocr
import json
 
from datetime import date
from dotenv import load_dotenv

class GenerationModel:

    def __init__(self) -> None:
        env_path = Path(__file__).parent / "API_key.env"
        load_dotenv(env_path)

        self.api_key = os.getenv("GEMINI_API_KEY")
        print("Loaded API KEY:", self.api_key)

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
        
        try:
            with open("Database/data.json", "r") as f:
                data_stored = json.load(f)
                total_entries = len(data_stored)
        
        except FileNotFoundError:
            data_stored = {}
            total_entries=0
        index = "DOC_" + "0"*(3-len(str((total_entries+1)))) + str(total_entries+1)
        data_entry = \
        {
            index:
            {
                "filename" : os.path.basename(file),
                "text" : self.text,
                "summary" : self.summary,
                "date" : f"{date.today()}",
                "category" : self.category,
                "vector_id" : index
            }
        }


        data_stored[index] = data_entry[index]

        with open("Database/data.json", "w") as f:
            json.dump(data_stored, f, indent=4)

        return data_stored[index]
    
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
        
        prompt_summary = f"""You are an AI engine designed for document intelligence. Summarize the provided document into a clear and concise textual summary.

                            RULES:
                            1. Output must be plain text only.
                            2. Do not use any asterisks, special characters, markdown formatting, or bullet points.
                            3. Do not add headings or labels.
                            4. Preserve the original meaning while shortening the content.
                            5. Ensure the summary is factual, coherent, and stays within the context of the document.

                            Now summarize the following document:
                            {text}
                        """
        summary = self.model.model.generate_content(prompt_summary)

        prompt_category = f"""You are an AI classifier for the Yellow Ranger Doc-Sage Intelligence Engine.
                            Your job is to classify the document into exactly one of the following predefined categories:
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
                            
                            RULES:
                            1. Output strictly one word only, exactly matching one of the categories above.
                            2. Output must be plain text only.
                            3. Do not output any symbols, punctuation, explanations, or phrases.
                            4. If the document partially fits many categories, choose the closest and most dominant one.
                            
                            Classify this document:
                            {text}
                        """
        category = self.model.model.generate_content(prompt_category)

        return summary.text, category.text
        

if __name__ == "__main__":

    extracter = TextExtracter()


import pymupdf
import os
import google.generativeai as genai

from dotenv import load_dotenv

class GenerationModel:

    def __init__(self) -> None:

        load_dotenv("API_key.env")
        self.api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")  

class TextExtracter:

    def __init__(self) -> None:

        self.model = GenerationModel()
            
    def extractText(self, filePath:str) -> str:
        
        with pymupdf.open(filePath) as doc:

            text = chr(12).join([page.get_text() for page in doc])
        
        return text

    def getSummary(self, text:str) -> str:

        prompt = ""
        response = self.model.generate_content(prompt)

        return response.text
    

if __name__ == "__main__":

    extracter = TextExtracter()


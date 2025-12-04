import pymupdf

class TextExtracter:

    def __init__(self) -> None:
        ...
    
    def extractText(self, filePath:str) -> str:
        
        with pymupdf.open(filePath) as doc:

            text = chr(12).join([page.get_text() for page in doc])
        
        return text

if __name__ == "__main__":

    extracter = TextExtracter()

    text = extracter.extractText(filePath=r'C:\Users\shrey\Downloads\ASANA.pdf')
    print(text)

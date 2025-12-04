from Backend.Extracter import TextExtracter

Extracter = TextExtracter()

for i in Extracter.handleFiles(r"C:/Users/shrey/Downloads/yellow_ranger_dummy_document.pdf"):
    print(i)
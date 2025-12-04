from Backend.Extracter import TextExtracter

Extracter = TextExtracter()

output = Extracter.handleFiles(r"C:/Users/shrey/Downloads/yellow_ranger_dummy_document.pdf")

print(output["summary"])

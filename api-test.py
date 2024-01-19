import fitz  # PyMuPDF
import requests

def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(start_page - 1, end_page):  # Page numbers are 0-based in PyMuPDF
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

pdf_file_path = r"./eks-ug.pdf"
start_page = 90
end_page = 95

extracted_text = extract_text_from_pdf(pdf_file_path, start_page, end_page)
print("length of the test", len(extracted_text))

API_URL = "http://dpo.asuscomm.com:8088/predict"

def query(payload):
    response = requests.post(API_URL, json=payload)
    # print(response.text)  # Add this line to print the raw response text
    return response.json()

output = query({
    "inputs": extracted_text,
})

print(output)
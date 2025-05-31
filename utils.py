from pypdf import PdfReader
import docx
import pandas as pd
import json 

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2)
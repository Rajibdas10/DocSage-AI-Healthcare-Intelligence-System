import os
import json
import pandas as pd
import docx
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF using PyMuPDF for better accuracy.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def extract_text_from_docx(file_path):
    """
    Extract text from DOCX files.
    """
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"

def extract_text_from_excel(file_path):
    """
    Extract text from Excel files, handling multiple sheets and large data.
    """
    try:
        text = ""
        excel = pd.ExcelFile(file_path)
        for sheet_name in excel.sheet_names:
            df = excel.parse(sheet_name)
            df = df.dropna(how='all').dropna(axis=1, how='all')
            sheet_text = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')
            text += f"Sheet: {sheet_name}\n{sheet_text}\n\n"
            if len(text) > 50000:  # Limit to prevent excessive memory usage
                break
        return text
    except Exception as e:
        return f"Error extracting text from Excel: {e}"

def extract_text_from_csv(file_path):
    """
    Extract text from CSV files.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error extracting text from CSV: {e}"

def extract_text_from_json(file_path):
    """
    Extract text from JSON files.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error extracting text from JSON: {e}"

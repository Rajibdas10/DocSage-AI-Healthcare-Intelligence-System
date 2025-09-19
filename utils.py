import os
import json
import pandas as pd
import docx
import fitz  # PyMuPDF
import gc

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF using PyMuPDF with memory management.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        page_count = 0
        
        for page in doc:
            # FIXED: Limit pages to prevent memory overflow
            if page_count >= 50:  # Process max 50 pages
                text += "\n... [PDF truncated - processed first 50 pages only]"
                break
                
            page_text = page.get_text()
            text += page_text
            page_count += 1
            
            # FIXED: Limit total text size
            if len(text) > 80000:  # 80KB limit
                text += "\n... [PDF truncated for memory management]"
                break
        
        doc.close()
        # FIXED: Memory cleanup
        gc.collect()
        return text
        
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def extract_text_from_docx(file_path):
    """
    Extract text from DOCX files with memory management.
    """
    try:
        doc = docx.Document(file_path)
        text_parts = []
        char_count = 0
        
        for para in doc.paragraphs:
            para_text = para.text
            if char_count + len(para_text) > 80000:  # FIXED: 80KB limit
                text_parts.append("\n... [Document truncated for memory management]")
                break
            text_parts.append(para_text)
            char_count += len(para_text)
        
        # FIXED: Memory cleanup
        result = "\n".join(text_parts)
        del doc, text_parts
        gc.collect()
        return result
        
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"

def extract_text_from_excel(file_path, save_json=False, json_path="code_description.json"):
    """
    Extract ICD code mappings from Excel with structured formatting.
    Also optionally save them as JSON for exact/fuzzy matching.
    """
    try:
        text = ""
        df = pd.read_excel(file_path, nrows=2000)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        code_map = {}

        if "CODE" in df.columns and any("DESCRIPTION" in c.upper() for c in df.columns):
            # Find the description column dynamically
            desc_col = next((c for c in df.columns if "DESCRIPTION" in c.upper()), None)
            if desc_col:
                df = df[["CODE", desc_col]].astype(str)
                df[desc_col] = df[desc_col].str.strip().str.slice(0, 200)
                df["combined"] = df["CODE"].str.upper() + " – " + df[desc_col]
                text = "\n".join(df["combined"].tolist())

                # Build code map
                for _, row in df.iterrows():
                    code_map[row["CODE"].strip().upper()] = row[desc_col].strip()

        else:
            # Fallback: generic text extraction
            df = df.astype(str).apply(lambda x: x.str[:100])
            text = df.apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')

        if save_json and code_map:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(code_map, f, indent=2, ensure_ascii=False)
            print(f"[✔] Saved code-description map to {json_path}")

        if len(text) > 50000:
            text = text[:50000] + "\n... [Excel truncated]"

        gc.collect()
        return text

    except Exception as e:
        return f"Error extracting text from Excel: {e}"


def extract_text_from_csv(file_path):
    """
    Extract ICD code mappings from CSV with structured formatting.
    """
    try:
        df = pd.read_csv(file_path, nrows=2000)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        if "CODE" in df.columns and any("DESCRIPTION" in c.upper() for c in df.columns):
            # Find the description column dynamically
            desc_col = next((c for c in df.columns if "DESCRIPTION" in c.upper()), None)
            if desc_col:
                df = df[["CODE", desc_col]].astype(str)
                df[desc_col] = df[desc_col].str.slice(0, 200)
                df["combined"] = df["CODE"] + " – " + df[desc_col]
                text = "\n".join(df["combined"].tolist())
            else:
                df = df.astype(str).apply(lambda x: x.str[:100])
                text = df.apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')
        else:
            df = df.astype(str).apply(lambda x: x.str[:100])
            text = df.apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')

        if len(text) > 50000:
            text = text[:50000] + "\n... [CSV truncated]"

        gc.collect()
        return text

    except Exception as e:
        return f"Error extracting text from CSV: {e}"


def extract_text_from_json(file_obj):
    """
    Extract text from JSON files with memory management.
    """
    try:
        if hasattr(file_obj, 'read'):
            data = json.load(file_obj)
        else:
            with open(file_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # FIXED: Limit JSON output size
        json_str = json.dumps(data, indent=2)
        
        if len(json_str) > 60000:  # 60KB limit
            # Try to truncate smartly
            if isinstance(data, dict):
                # Keep only first few keys if it's a large dict
                limited_data = dict(list(data.items())[:10])
                json_str = json.dumps(limited_data, indent=2)
                json_str += "\n... [JSON truncated - showing first 10 keys only]"
            elif isinstance(data, list):
                # Keep only first few items if it's a large list
                limited_data = data[:50]
                json_str = json.dumps(limited_data, indent=2)
                json_str += "\n... [JSON truncated - showing first 50 items only]"
            else:
                json_str = json_str[:60000] + "\n... [JSON truncated for memory management]"
        
        # FIXED: Memory cleanup
        del data
        gc.collect()
        return json_str
        
    except Exception as e:
        return f"Error extracting text from JSON: {e}"
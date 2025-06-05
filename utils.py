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

def extract_text_from_excel(file_path):
    """
    Extract text from Excel files, handling multiple sheets with strict memory limits.
    """
    try:
        text = ""
        excel = pd.ExcelFile(file_path)
        sheets_processed = 0
        
        for sheet_name in excel.sheet_names:
            # FIXED: Limit number of sheets
            if sheets_processed >= 3:  # Process max 3 sheets
                text += f"\n... [Remaining sheets skipped for memory management]"
                break
                
            try:
                # FIXED: Read with row limit
                df = excel.parse(sheet_name, nrows=1000)  # Max 1000 rows per sheet
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                # FIXED: Limit columns processed
                if len(df.columns) > 20:
                    df = df.iloc[:, :20]  # First 20 columns only
                
                # FIXED: Truncate cell content
                df = df.astype(str).apply(lambda x: x.str[:100])  # Max 100 chars per cell
                
                sheet_text = df.apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')
                text += f"Sheet: {sheet_name}\n{sheet_text}\n\n"
                sheets_processed += 1
                
                # FIXED: Check total size
                if len(text) > 40000:  # 40KB limit for Excel
                    text += "\n... [Excel file truncated for memory management]"
                    break
                    
            except Exception as sheet_error:
                text += f"Sheet: {sheet_name}\nError processing sheet: {sheet_error}\n\n"
                
        excel.close()
        # FIXED: Memory cleanup
        gc.collect()
        return text
        
    except Exception as e:
        return f"Error extracting text from Excel: {e}"

def extract_text_from_csv(file_path):
    """
    Extract text from CSV files with memory management.
    """
    try:
        # FIXED: Read with limits
        df = pd.read_csv(file_path, nrows=2000)  # Max 2000 rows
        
        # FIXED: Limit columns
        if len(df.columns) > 15:
            df = df.iloc[:, :15]  # First 15 columns only
            
        # FIXED: Truncate cell content
        df = df.astype(str).apply(lambda x: x.str[:100])  # Max 100 chars per cell
        
        result = df.to_string(index=False, max_rows=1500)  # Limit output
        
        # FIXED: Check size and truncate if needed
        if len(result) > 50000:
            result = result[:50000] + "\n... [CSV truncated for memory management]"
            
        # FIXED: Memory cleanup
        del df
        gc.collect()
        return result
        
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
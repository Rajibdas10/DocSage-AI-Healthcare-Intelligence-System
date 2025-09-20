import json
import re
from rapidfuzz import fuzz

# Load dictionary once
try:
    with open("code_description.json", "r", encoding="utf-8") as f:
        code_map = json.load(f)
except FileNotFoundError:
    code_map = {}
    print("[Warning] code_description.json not found. Fuzzy matching will be skipped.")

def is_icd_code(text):
    """Check if the input is an ICD-style code (e.g., A0223)."""
    return re.match(r"^[A-Z]\d{2,4}$", text.strip()) is not None

def search_code_exact(query):
    """Exact match lookup from local dictionary."""
    query = query.strip().upper()
    return code_map.get(query)

def search_description_fuzzy(query, threshold=85):
    """
    Fuzzy match a description to find the closest code.
    Lowered threshold slightly to improve partial matches.
    """
    best_match = None
    best_score = 0
    for code, desc in code_map.items():
        score = fuzz.partial_ratio(query.lower(), desc.lower())
        if score > best_score and score >= threshold:
            best_match = (code, desc)
            best_score = score
    return best_match

def handle_query(query, vector_search_fn=None, top_k=3):
    """
    Hybrid search:
    1. If query is a code -> get from local JSON first.
    2. If query is a description -> try fuzzy match, then fallback to vector search.
    vector_search_fn should be a callable(query, top_k) returning list of matches.
    """
    query = query.strip()

    # Case 1: User gave a code (e.g., "A0223")
    if is_icd_code(query):
        desc = search_code_exact(query)
        if desc:
            return {"type": "code_lookup", "code": query, "description": desc}
        # Fallback to vector search if not found locally
        if vector_search_fn:
            return {"type": "vector_fallback", "results": vector_search_fn(query, top_k)}
        return {"error": f"Code {query} not found."}

    # Case 2: User gave a description (e.g., "Salmonella arthritis")
    match = search_description_fuzzy(query)
    if match:
        code, desc = match
        return {"type": "description_lookup", "code": code, "description": desc}

    # Fallback to vector DB if available
    if vector_search_fn:
        return {"type": "vector_fallback", "results": vector_search_fn(query, top_k)}

    return {"error": "No match found."}

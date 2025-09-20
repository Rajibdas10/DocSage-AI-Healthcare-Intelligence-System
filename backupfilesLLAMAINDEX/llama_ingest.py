import os
import re
import json
import chromadb
from chromadb.config import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from utils import (
    extract_text_from_excel,
    extract_text_from_csv,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_json
)

# ✅ Save extracted code-description map to JSON
def save_code_description_map(code_map, output_path="code_description.json"):
    if code_map:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(code_map, f, indent=2, ensure_ascii=False)
        print(f"[✔] Saved code-description map to {output_path}")
    else:
        print("[!] No code-description pairs found. JSON not created.")

# ✅ Main ingestion function
def ingest_docs(file_path, persist_dir="chroma_store"):
    print(f"\n[Ingesting] File: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()

    docs = []
    code_map = {}
    text_data = ""

    # --- Handle different file types ---
    if ext in [".xlsx", ".xls"]:
        print("[Type] Excel detected")
        text_data = extract_text_from_excel(file_path, save_json=True)
    elif ext == ".csv":
        print("[Type] CSV detected")
        text_data = extract_text_from_csv(file_path)
        # We extract code map from CSV text using regex
        code_map = extract_code_description_pairs_from_text(text_data)
        save_code_description_map(code_map)
    elif ext == ".pdf":
        print("[Type] PDF detected")
        text_data = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        print("[Type] DOCX detected")
        text_data = extract_text_from_docx(file_path)
    elif ext == ".json":
        print("[Type] JSON detected")
        text_data = extract_text_from_json(file_path)
    else:
        print("[Type] Default text loader")
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()

    # --- Convert text_data into LlamaIndex docs ---
    if text_data and not docs:
        from llama_index.core.schema import Document
        docs = [Document(text=text_data)]

    # Preview first 300 chars
    if docs:
        print("\n[Preview] First 300 chars of document:")
        print(docs[0].text[:300])

    # --- Build and store vector index ---
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    collection_name = "documents"
    try:
        chroma_client.delete_collection(collection_name)
        print("[Cleanup] Deleted old collection")
    except:
        pass

    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    splitter = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=100)
    print(f"[Settings] Chunking: line-aware, chunk_size=800, overlap=100")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter]
    )

    print("[Done] Ingestion complete and index stored.\n")
    return index

# ✅ Helper for CSV/text regex extraction
def extract_code_description_pairs_from_text(text):
    pattern = r"^([A-Z]\d{2,4})\s*[–-]\s*(.+)$"
    code_map = {}
    for line in text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            code, description = match.groups()
            code_map[code.strip().upper()] = description.strip()
    return code_map

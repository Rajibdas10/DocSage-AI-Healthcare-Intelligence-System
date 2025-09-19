
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DB_DIR = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  
    )

def save_to_chroma(docs, persist_dir=CHROMA_DB_DIR):
    embeddings = get_embeddings()
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    db.persist()
    return db

def load_chroma(persist_dir=CHROMA_DB_DIR):
    embeddings = get_embeddings()
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return None

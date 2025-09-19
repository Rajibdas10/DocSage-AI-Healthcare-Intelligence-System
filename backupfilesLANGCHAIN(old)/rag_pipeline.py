from langchain.text_splitter import CharacterTextSplitter
from vectorstore import save_to_chroma, load_chroma
import spacy
import os

# Load SpaCy once
nlp = spacy.load("en_core_web_sm")

def create_chunks(text, metadata=None):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    if metadata:
        for c in chunks:
            c.metadata = metadata
    return chunks

def build_or_update_vectorstore(text, filename):
    metadata = {"source": filename}
    chunks = create_chunks(text, metadata)
    db = save_to_chroma(chunks)
    return db

def extract_keywords_from_query(query):
    doc = nlp(query)
    keywords = list(set([ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2]))
    return keywords

def query_rag(query, db, top_k=6, keyword=None):
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    # NEW: extract keywords dynamically
    if keyword is None:
        keyword_list = extract_keywords_from_query(query)
    else:
        keyword_list = [keyword]

    # NEW: dynamic filter based on keyword list
    if keyword_list:
        filtered_docs = []
        for doc in docs:
            for key in keyword_list:
                if key.lower() in doc.page_content.lower():
                    filtered_docs.append(doc)
                    break  # Avoid duplicates
        if filtered_docs:
            docs = filtered_docs

    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs

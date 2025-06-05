import os
import tempfile
import warnings
import gc
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variables for Streamlit
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_WATCH_FILE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
# Set HOME directory to a writable location
home_dir = tempfile.mkdtemp()
os.environ["HOME"] = home_dir

import io
import base64
import streamlit as st
from dotenv import load_dotenv
import requests
import json

from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_excel, extract_text_from_csv, extract_text_from_json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  

# Streamlit page config
st.set_page_config(page_title="MedValidate AI", layout="wide")
st.title("🏥 Alles Health AI")
st.markdown("Upload your **Tariff Document** and ask reimbursement-related questions.")

# Initialize session state variables
if "processed_content" not in st.session_state:
    st.session_state.processed_content = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "processing" not in st.session_state:
    st.session_state.processing = False

# API Key setup - FIXED: Check for multiple possible environment variable names
load_dotenv()
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY") or 
    os.getenv("api") or  # Your HuggingFace secret name
    st.secrets.get("GROQ_API_KEY") or 
    st.secrets.get("api")
)

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY or 'api' in your environment variables or Streamlit secrets.")
    st.stop()

@st.cache_resource
def load_embeddings():
    """Load embeddings model - cached to avoid reloading"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def query_groq_api(prompt, max_tokens=200):
    """Query Groq API with error handling and memory management"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # FIXED: Limit prompt size to prevent buffer overflow
    if len(prompt) > 3000:
        prompt = prompt[:3000] + "... [truncated for processing]"
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a medical insurance expert specializing in reimbursement queries. Provide accurate, concise answers based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # FIXED: Memory cleanup after API call
        answer = result["choices"][0]["message"]["content"].strip()
        del result, response
        gc.collect()
        
        return answer
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def process_document(uploaded_file):
    """Process uploaded document and extract content with size limits"""
    if uploaded_file is None:
        return None
    
    # FIXED: Add file size check
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Please upload a file smaller than 10MB.")
        return None
    
    # Create a unique key for this file to cache processing
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Check if we already processed this exact file
    if hasattr(st.session_state, 'last_file_key') and st.session_state.last_file_key == file_key:
        return st.session_state.processed_content
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    tmp_path = tmp_file.name

    try:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()

        # Process based on file type
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(tmp_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = extract_text_from_docx(tmp_path)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            content = extract_text_from_excel(tmp_path)
        elif uploaded_file.type == "text/csv":
            content = extract_text_from_csv(tmp_path)
        elif uploaded_file.type == "application/json":
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = extract_text_from_json(f)
        else:
            st.error("Unsupported file format")
            return None

        # FIXED: Limit content size to prevent memory issues
        if content and len(content) > 100000:  # 100KB text limit
            content = content[:100000] + "\n... [Document truncated for processing]"
            st.warning("Document was truncated to prevent memory issues. Processing first 100KB only.")

        # Cache the processed content
        st.session_state.processed_content = content
        st.session_state.last_file_key = file_key
        st.success("✅ Document processed successfully")
        
        # FIXED: Memory cleanup
        gc.collect()
        
        return content
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass

def process_query(content, query):
    """Process the query against the document content with memory management"""
    if not content or not query.strip():
        return None, []
    
    try:
        embeddings = load_embeddings()
        
        # FIXED: Smaller chunk size to prevent memory issues
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([content])

        # FIXED: Limit number of documents to prevent memory overflow
        if len(docs) > 20:
            docs = docs[:20]
            st.info("Processing limited to first 20 document chunks to prevent memory issues.")

        # Create vector store
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})  # FIXED: Reduced from 3 to 2
        
        # Use invoke instead of deprecated get_relevant_documents
        rel_docs = retriever.invoke(query)

        # FIXED: Limit combined context size more strictly
        combined_context = ""
        for doc in rel_docs:
            if len(combined_context + doc.page_content) < 800:  # FIXED: Reduced from 1500
                combined_context += doc.page_content + "\n\n"
            else:
                break

        context = combined_context or "No relevant context found"

        # Create prompt with size limit
        prompt = f"""Based on the following medical tariff document context, answer the specific question about reimbursement codes or procedures.

CONTEXT:
{context[:600]}

QUESTION: {query}

Please provide a direct, specific answer focusing on medical codes, procedures, or reimbursement information. If the exact information isn't in the context, state what related information is available."""

        # Query API
        answer = query_groq_api(prompt, max_tokens=150)  # FIXED: Reduced token limit
        
        # FIXED: Memory cleanup
        del db, retriever, embeddings, docs
        gc.collect()
        
        return answer, rel_docs
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        # Memory cleanup on error
        gc.collect()
        return None, []

# FIXED: Add periodic memory cleanup
def cleanup_memory():
    """Cleanup memory periodically"""
    if len(st.session_state.keys()) > 15:
        # Keep only essential session state
        essential_keys = ['processed_content', 'last_file_key', 'current_query', 'answer', 'last_processed_query']
        for key in list(st.session_state.keys()):
            if key not in essential_keys:
                del st.session_state[key]
    gc.collect()

# Main UI
uploaded_file = st.file_uploader(
    "Upload Tariff Document (PDF, Word, Excel, CSV, JSON)", 
    type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'json']
)

# Process document when uploaded
if uploaded_file:
    content = process_document(uploaded_file)
    st.session_state.processed_content = content

# Query input
query = st.text_input(
    "💬 Ask Your Reimbursement Question", 
    value="",
    key="query_input"
)

# Process query button
if st.button("🔍 Get Answer", disabled=st.session_state.processing):
    if not st.session_state.processed_content:
        st.error("Please upload a document first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        # Only process if it's a new query or content has changed
        if query != st.session_state.last_processed_query:
            st.session_state.processing = True
            st.session_state.current_query = query
            
            with st.spinner("Processing your question..."):
                try:
                    answer, rel_docs = process_query(st.session_state.processed_content, query)
                    
                    if answer and len(answer) > 20:
                        st.session_state.answer = answer
                        st.session_state.rel_docs = rel_docs
                        st.session_state.last_processed_query = query
                    else:
                        st.session_state.answer = "Could not generate a satisfactory answer. Please try rephrasing your question."
                        st.session_state.rel_docs = []
                        
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.session_state.answer = "Processing failed. Please try again with a simpler question."
                    st.session_state.rel_docs = []
            
            st.session_state.processing = False
            # FIXED: Cleanup after processing
            cleanup_memory()

# Display results
if st.session_state.answer and st.session_state.current_query:
    st.markdown("### ✅ Answer:")
    st.markdown(st.session_state.answer)
    
    # Prepare download content (FIXED: Limit size)
    if hasattr(st.session_state, 'rel_docs') and st.session_state.rel_docs:
        output_text = f"Q: {st.session_state.current_query}\n\nA: {st.session_state.answer}\n\nSources:\n"
        for i, doc in enumerate(st.session_state.rel_docs[:2]):  # FIXED: Limit to 2 docs
            output_text += f"\nChunk {i+1}:\n{doc.page_content[:500]}\n"  # FIXED: Limit chunk size
        
        # FIXED: Check output size before encoding
        if len(output_text) < 50000:  # 50KB limit
            b64 = base64.b64encode(output_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="response.txt">📄 Download Answer</a>'
            st.markdown(href, unsafe_allow_html=True)

# Clear/Reset functionality
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Ask New Question"):
        st.session_state.current_query = ""
        st.session_state.answer = ""
        st.session_state.last_processed_query = ""
        if hasattr(st.session_state, 'rel_docs'):
            del st.session_state.rel_docs
        cleanup_memory()  # FIXED: Add cleanup
        st.rerun()

with col2:
    if st.button("🗑️ Clear All"):
        for key in list(st.session_state.keys()):
            if key not in ['processed_content', 'last_file_key']:  # Keep processed document
                del st.session_state[key]
        cleanup_memory()  # FIXED: Add cleanup
        st.rerun()

# Show processing status
if st.session_state.processing:
    st.info("⏳ Processing... Please wait.")

# FIXED: Display memory usage info for debugging
if st.checkbox("Show Debug Info"):
    st.write(f"Session state keys: {len(st.session_state.keys())}")
    if st.session_state.processed_content:
        st.write(f"Document size: {len(st.session_state.processed_content)} characters")
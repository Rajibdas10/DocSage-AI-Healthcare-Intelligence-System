import io
import base64
import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables
load_dotenv()

from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_excel, extract_text_from_csv, extract_text_from_json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile

# Streamlit app setup
st.set_page_config(page_title="MedValidate AI", layout="wide")

st.title("🏥 Alles Health AI")
st.markdown("Upload your **Tariff Document** and ask reimbursement-related questions.")

# Get Groq API key from environment or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in your environment variables or Streamlit secrets. Get free API key from: https://groq.com")
    st.stop()

uploaded_file = st.file_uploader("Upload Tariff Document (PDF, Word, Excel, CSV, JSON)", type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'json'])
query = st.text_input("💬 Ask Your Reimbursement Question")

@st.cache_resource  
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def query_groq_api(prompt, max_tokens=200):
    """Query Groq API - very fast and free tier available"""
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",  # Fast Llama3 model
        "messages": [
            {
                "role": "system",
                "content": "You are a medical insurance expert specializing in reimbursement queries. Provide accurate, concise answers based on the provided context."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return None

if uploaded_file:
    # Create temp file with proper cleanup
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    tmp_path = tmp_file.name
    
    try:
        # Write file content
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()  # Important: Close file before processing
        
        # Process the file
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(tmp_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = extract_text_from_docx(tmp_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or uploaded_file.type == "application/vnd.ms-excel":
            content = extract_text_from_excel(tmp_path)
        elif uploaded_file.type == "text/csv":
            content = extract_text_from_csv(tmp_path)
        elif uploaded_file.type == "application/json":
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = extract_text_from_json(f)
        else:
            content = "Unsupported format"
        
        st.success("✅ Document processed")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        content = None
    
    finally:
        # Clean up temp file with better error handling
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except PermissionError:
            # If we can't delete immediately, try again after a short delay
            import time
            time.sleep(0.1)
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore if still can't delete
    
    # LLM logic
    if query and content:
        st.info("🔍 Validating query...")
        
        try:
            # Load embeddings
            embeddings = load_embeddings()
            
            # Split and embed
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([content])
            
            # Create vector store
            with st.spinner("Processing document..."):
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 3})
                rel_docs = retriever.get_relevant_documents(query)
            
            # Combine context
            combined_context = ""
            for doc in rel_docs:
                if len(combined_context + doc.page_content) < 1500:
                    combined_context += doc.page_content + "\n\n"
                else:
                    break
            
            context = combined_context if combined_context else "No relevant context found"
            
            # Create optimized prompt
            prompt = f"""Based on the following medical tariff document context, answer the specific question about reimbursement codes or procedures.

CONTEXT:
{context[:1000]}

QUESTION: {query}

Please provide a direct, specific answer focusing on medical codes, procedures, or reimbursement information. If the exact information isn't in the context, state what related information is available."""
            
            with st.spinner("Generating answer..."):
                answer = query_groq_api(prompt, max_tokens=200)
                
                if answer and len(answer) > 20:
                    st.markdown("### ✅ Answer:")
                    st.markdown(answer)

                    # ✅ Downloadable response block
                    output_text = f"Q: {query}\n\nA: {answer}\n\nSources:\n"
                    for i, doc in enumerate(rel_docs):
                        output_text += f"\nChunk {i+1}:\n{doc.page_content}\n"

                    b64 = base64.b64encode(output_text.encode()).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="response.txt">📄 Download Answer</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Fallback to showing relevant context
                    st.markdown(f"### 📋 Relevant Information Found:")
                    st.text(context[:800] + "..." if len(context) > 800 else context)
                    
        except Exception as e:
            st.error(f"Error in processing: {e}")
            # Final fallback
            st.markdown("### 📄 Document Content Preview:")
            st.text(content[:1000] + "..." if len(content) > 1000 else content)
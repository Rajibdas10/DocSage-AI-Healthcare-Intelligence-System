from utils import extract_text_from_excel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import gc

# === Load embeddings model ===
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

def create_vector_store_from_text(content):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([content])
    db = FAISS.from_documents(docs, embeddings)
    return db

def hybrid_query(query, db, keyword=None, top_k=3):
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    if keyword:
        keyword = keyword.lower()
        filtered = [doc for doc in docs if keyword in doc.page_content.lower()]
        if filtered:
            docs = filtered

    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs

# === Sample ICD QA Test Cases ===
test_queries = [
    ("What is the ICD-10 code for Typhoid pneumonia?", "A0103"),
    ("List all ICD-10 codes related to Salmonella infections.", "A020"),
    ("What does code A0105 refer to?", "Typhoid osteomyelitis"),
    ("Is there a code for Shigellosis due to Shigella boydii?", "A033"),
    ("Give all ICD-10 codes for paratyphoid fever.", "A011"),
]

def test_rag_accuracy(content):
    db = create_vector_store_from_text(content)
    print("\n===== ICD-10 Query Accuracy Test =====")
    passed = 0

    for i, (query, expected) in enumerate(test_queries, 1):
        context, docs = hybrid_query(query, db)
        hit = expected.lower() in context.lower()
        result = "✅ PASS" if hit else "❌ FAIL"
        print(f"Q{i}: {query}\nExpected: {expected}\nContext: {context[:150]}...\nResult: {result}\n")
        passed += hit
        gc.collect()

    print(f"🎯 Accuracy Score: {passed}/{len(test_queries)} passed.")

if __name__ == "__main__":
    # Replace with your actual file path if not in root
    content = extract_text_from_excel("testfiles/ICD 10 - Codes_Jan2025.xlsx")
    test_rag_accuracy(content)

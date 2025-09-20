from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import chromadb
from chromadb.config import Settings
import os

def load_index(persist_dir="chroma_store"):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    collection_name = "documents"
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except ValueError:
        chroma_collection = chroma_client.create_collection(collection_name)

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
        host=None,
        port=None
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    return index

def query_llama(query_text):
    try:
        index = load_index()

        llm = Groq(
            model="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )

        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=5,  # ✅ Increased from 2 → 5
            response_mode="tree_summarize"
        )

        # ✅ DEBUG: Print retrieved chunks for transparency
        print("\n[Debug] Retrieved Chunks:")
        for i, node in enumerate(query_engine.retrieve(query_text)):
            print(f"\n--- Chunk {i+1} ---\n")
            print(node.get_text())

        response = query_engine.query(query_text)
        return str(response)

    except Exception as e:
        return f"❌ Error: {str(e)}"

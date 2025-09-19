import os
import uuid
import tempfile
import re
from typing import List, Dict, Any, Optional
import logging
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix PyTorch compatibility issues
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# ChromaDB
import chromadb
from chromadb.config import Settings as ChromaSettings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareRAGPipeline:
    """
    Complete RAG Pipeline for Healthcare Documents following the NLP workflow:
    1. File Ingestion → 2. Text Extraction → 3. Preprocessing → 4. Chunking
    → 5. Embedding Generation → 6. Vector Storage → 7. Query Processing
    → 8. Similarity Search → 9. Context Assembly → 10. LLM Response → 11. Return Result
    """

    def __init__(self, 
                 groq_api_key: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "llama-3.3-70b-versatile",
                 chunk_size: int = 256,
                 chunk_overlap: int = 20):
        """
        Initialize the RAG pipeline with proper models and settings
        """
        self.groq_api_key = groq_api_key
        self.embedding_model_name = embedding_model
        self.llm_model_name= llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Supported models for embedding 
        self.embedding_models ={
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",

        }
        # Supported models for LLM
        self.llm_models ={
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant": "llama-3.1-8b-instant",
            "openai/gpt-oss-20b" : "openai/gpt-oss-20b",              
        } 
        # Validate embedding model 
        if self.embedding_model_name not in self.embedding_models:
            raise ValueError(f"Embedding model '{self.embedding_model_name}' not supported")

         # Validate LLM model
        if self.llm_model_name not in self.llm_models:
            raise ValueError(f"LLM model '{self.llm_model_name}' not supported.")

        # Log current configuration
        logger.info(f"🔧 Initializing RAG Pipeline with:")
        logger.info(f"   - Embedding Model: {self.embedding_model_name}")
        logger.info(f"   - LLM Model: {self.llm_model_name}")
        logger.info(f"   - Chunk Size: {self.chunk_size}")
        logger.info(f"   - Chunk Overlap: {self.chunk_overlap}")

        # Setup models
        self._setup_models()

        # Initialize storage
        self.store_dir = None
        self.chroma_client = None
        self.vector_store = None
        self.index = None

    def _setup_models(self):
        """Setup embedding model and LLM with proper configuration"""
        try:
            # Setup embedding model (Step 5: Embedding Generation)
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_models[self.embedding_model_name],
                trust_remote_code=True
            )

            # Setup Groq LLM (Step 11: LLM Response)
            logger.info(f"Setting up Groq LLM: {self.llm_model_name}")
            self.llm = Groq(
                model=self.llm_models[self.llm_model_name],
                api_key=self.groq_api_key,
                temperature=0.1  # Lower temperature for more consistent healthcare responses
            )

            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model

            logger.info("✅ Models setup complete")

        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise

    def get_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration"""
        return {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    def preprocess_text(self, text: str) -> str:
        """
        Step 3: Preprocessing - Clean and prepare text for chunking
        """
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove special characters but keep medical terminology
            text = re.sub(r'[^\w\s\.\-\(\)\[\]\,\;\:]', ' ', text)

            # Remove excessive newlines
            text = re.sub(r'\n+', '\n', text)

            # Strip and ensure we have content
            text = text.strip()

            logger.info(f"Text preprocessing complete. Length: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text

    def enhance_medical_text(self, text: str) -> str:
        """Enhance medical text for better retrieval"""
        lines = text.strip().split('\n')
        enhanced_lines = []
        
        for line in lines:
            if line.strip():
                # Handle CSV/Excel data format
                parts = line.split('\t') if '\t' in line else line.split(',', 1)
                if len(parts) >= 2:
                    code = parts[0].strip()
                    description = parts[1].strip()
                    enhanced_line = f"ICD-10 Code {code}: {description}. Medical condition: {description}"
                else:
                    enhanced_line = f"Medical Code Entry: {line.strip()}"
                
                enhanced_lines.append(enhanced_line)
        
        return '\n'.join(enhanced_lines)

    def extract_medical_codes(self, text: str) -> List[str]:
        """Extract ICD-10 codes from text"""
        import re
        pattern = r'\b[A-Z]\d{2,3}(?:\.\d{1,2})?\b'
        return re.findall(pattern, text)

    def extract_conditions(self, text: str) -> List[str]:
        """Extract medical condition names"""
        conditions = []
        lines = text.split('\n')
        for line in lines:
            if ':' in line:
                condition = line.split(':', 1)[1].strip()
                if condition:
                    conditions.append(condition)
        return conditions

    def enhance_medical_query(self, query: str) -> str:
        """Enhance queries for better medical code retrieval"""
        
        medical_synonyms = {
            'fever': ['fever', 'pyrexia', 'hyperthermia'],
            'infection': ['infection', 'sepsis', 'bacterial', 'viral'],
            'typhoid': ['typhoid', 'typhoid fever', 'salmonella typhi'],
            'cholera': ['cholera', 'vibrio cholerae'],
            'salmonella': ['salmonella', 'salmonellosis'],
            'pneumonia': ['pneumonia', 'lung infection'],
            'arthritis': ['arthritis', 'joint inflammation'],
            'meningitis': ['meningitis', 'brain inflammation']
        }
        
        enhanced_query = query.lower()
        
        # Add medical context and synonyms
        query_terms = enhanced_query.split()
        expanded_terms = []
        
        for term in query_terms:
            expanded_terms.append(term)
            for base_term, synonyms in medical_synonyms.items():
                if term in base_term or base_term in term:
                    expanded_terms.extend(synonyms[:2])  # Add first 2 synonyms
        
        # Create comprehensive query
        final_query = f"Find medical condition or ICD-10 code related to: {' '.join(set(expanded_terms))}"
        return final_query

    def post_process_medical_response(self, response: str, original_query: str) -> str:
        """Post-process response for medical accuracy"""
        
        if not any(keyword in response.lower() for keyword in ['code', 'icd', 'condition']):
            response = f"Based on the medical documentation: {response}"
        
        # Format ICD codes properly
        import re
        def format_code(match):
            return f"ICD-10 Code {match.group()}"
        
        response = re.sub(r'\b[A-Z]\d{2,3}(?:\.\d{1,2})?\b', format_code, response)
        return response

    def create_chunks(self, text: str) -> List[Document]:
        """
        Medical-optimized chunking for ICD-10 codes and descriptions
        """
        try:
            # Enhance text for medical context first
            enhanced_text = self.enhance_medical_text(text)
            clean_text = self.preprocess_text(enhanced_text)

            # Use smaller chunks for medical precision
            medical_chunk_size = min(self.chunk_size, 300)
            medical_overlap = max(self.chunk_overlap, 100)
            
            logger.info(f"Creating medical chunks with size={medical_chunk_size}, overlap={medical_overlap}")
            
            splitter = SentenceSplitter(
                chunk_size=medical_chunk_size,
                chunk_overlap=medical_overlap,
                paragraph_separator="\n",
                secondary_chunking_regex=r"[A-Z]\d{2,3}",
                tokenizer=lambda x: x.split()
            )

            text_nodes = splitter.get_nodes_from_documents([Document(text=clean_text)])

            documents = []
            for i, node in enumerate(text_nodes):
                medical_codes = self.extract_medical_codes(node.text)
                conditions = self.extract_conditions(node.text)
                
                # FIX: Convert lists to strings for ChromaDB compatibility
                doc = Document(
                    text=node.text,
                    metadata={
                        "chunk_id": f"medical_chunk_{i}",
                        "chunk_size": len(node.text),
                        "word_count": len(node.text.split()),
                        "source": "icd10_codes",
                        # Convert lists to comma-separated strings
                        "medical_codes": ", ".join(medical_codes) if medical_codes else "",
                        "conditions": ", ".join(conditions) if conditions else "",
                        "medical_codes_count": len(medical_codes),
                        "conditions_count": len(conditions),
                        "embedding_model": str(self.embedding_model_name),
                        "llm_model": str(self.llm_model_name)
                    }
                )
                documents.append(doc)

            logger.info(f"✅ Created {len(documents)} medical chunks")
            
            # Log chunk statistics
            chunk_sizes = [len(doc.text) for doc in documents]
            logger.info(f"Chunk size stats - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.1f}")

            return documents

        except Exception as e:
            logger.error(f"Error creating medical chunks: {e}")
            raise

    def setup_vector_store(self) -> tuple:
        """
        Step 6: Vector Storage - Setup ChromaDB with persistence
        """
        try:
            # Create unique store directory with model info
            store_id = uuid.uuid4().hex[:8]
            
            # Fix model info string generation
            llm_model_str = str(self.llm_model_name).replace('-', '_')
            model_info = f"{self.embedding_model_name.replace('/', '_')}_{llm_model_str}"

            self.store_dir = os.path.join(
                tempfile.gettempdir(), 
                f"healthcare_rag_{model_info}_{store_id}"
            )
            os.makedirs(self.store_dir, exist_ok=True)

            # Initialize ChromaDB with persistence
            chroma_settings = ChromaSettings(
                persist_directory=self.store_dir,
                anonymized_telemetry=False
            )

            self.chroma_client = chromadb.PersistentClient(
                path=self.store_dir,
                settings=chroma_settings
            )

            # Create or get collection with model-specific name
            collection_name = f"healthcare_docs_{self.embedding_model_name.replace('/', '_').replace('-', '_')}"
            
            # Fix metadata - ChromaDB only accepts simple string/number values, not dicts
            collection_metadata = {
                "description": "Healthcare document embeddings",
                "embedding_model": str(self.embedding_model_name),
                "llm_model": str(self.llm_model_name),
                "chunk_size": str(self.chunk_size),
                "chunk_overlap": str(self.chunk_overlap)
            }

            try:
                # Try to get existing collection first
                collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"✅ Using existing collection: {collection_name}")
            except Exception as get_error:
                logger.info(f"Collection {collection_name} not found, creating new one...")
                try:
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata=collection_metadata
                    )
                    logger.info(f"✅ Created new collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}")
                    # Fallback to a simple collection name without special characters
                    fallback_name = f"healthcare_docs_{store_id}"
                    logger.info(f"Trying fallback collection name: {fallback_name}")
                    collection = self.chroma_client.create_collection(
                        name=fallback_name,
                        metadata={"description": "Healthcare document embeddings"}
                    )

            # Setup LlamaIndex vector store
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            logger.info(f"✅ Vector store setup complete at: {self.store_dir}")
            return self.store_dir, storage_context

        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise

    def process_document(self, extracted_text: str) -> str:
        """
        Main processing pipeline: Text → Chunks → Embeddings → Vector Storage
        """
        try:
            logger.info("🚀 Starting RAG pipeline processing...")
            logger.info(f"Using configuration: {self.get_configuration()}")

            # Step 4: Create chunks
            documents = self.create_chunks(extracted_text)

            if not documents:
                raise ValueError("No chunks created from extracted text")

            # Step 6: Setup vector storage
            store_dir, storage_context = self.setup_vector_store()

            # Step 5 & 6: Generate embeddings and store in vector DB
            logger.info(f"Generating embeddings using {self.embedding_model_name}...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )

            logger.info("✅ Document processing complete!")
            return store_dir

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    def create_query_engine(self, store_dir: str = None, similarity_top_k: int = 10):
        """
        Create query engine optimized for medical data
        """
        try:
            if store_dir and store_dir != self.store_dir:
                self.store_dir = store_dir
                self._load_existing_index()

            if not self.index:
                raise ValueError("No index available. Process a document first.")

            logger.info(f"Creating medical query engine with top_k={similarity_top_k}")
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode="tree_summarize",  # Better for structured medical data
                verbose=True
            )

            logger.info(f"✅ Medical query engine created with top_k={similarity_top_k}")
            return query_engine

        except Exception as e:
            logger.error(f"Error creating query engine: {e}")
            raise
    
    def _load_existing_index(self):
        """Load existing index from store directory"""
        try:
            # Setup ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.store_dir)
            
            # List all collections to find the right one
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            # Try to find collection with model-specific name
            target_name = f"healthcare_docs_{self.embedding_model_name.replace('/', '_').replace('-', '_')}"
            
            collection = None
            if target_name in collection_names:
                collection = self.chroma_client.get_collection(target_name)
                logger.info(f"Found target collection: {target_name}")
            elif collection_names:
                # Use the first available collection
                collection = self.chroma_client.get_collection(collection_names[0])
                logger.info(f"Using fallback collection: {collection_names[0]}")
            else:
                raise ValueError("No collections found in the store directory")

            # Setup vector store and index
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model  # Use current embedding model
            )

            logger.info("✅ Existing index loaded successfully")

        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            raise

    def query_documents(self, query: str, store_dir: str = None, similarity_top_k: int = 10) -> Dict[str, Any]:
        """
        Step 7-12: Complete query processing pipeline
        """
        try:
            logger.info(f"Processing query with {self.llm_model_name}: {query[:100]}...")

            # Create query engine with current settings
            query_engine = self.create_query_engine(store_dir, similarity_top_k)

            # Process query (Steps 7-11: Query → Embedding → Search → Context → LLM → Response)
            response = query_engine.query(query)

            # Extract response details
            result = {
                "answer": str(response),
                "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                "metadata": getattr(response, 'metadata', {}),
                "query": query,
                "configuration": self.get_configuration()
            }

            logger.info("✅ Query processing complete")
            return result

        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise

# Convenience functions for backward compatibility
def create_rag_pipeline(groq_api_key: str, **kwargs) -> HealthcareRAGPipeline:
    """Create a new RAG pipeline instance with specified configuration"""
    return HealthcareRAGPipeline(groq_api_key=groq_api_key, **kwargs)

def process_document(extracted_text: str, groq_api_key: str = None, **kwargs) -> str:
    """Process document with specified or default settings"""
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

    pipeline = create_rag_pipeline(groq_api_key, **kwargs)
    return pipeline.process_document(extracted_text)

# Example usage and testing
if __name__ == "__main__":
    # Test the pipeline
    import os

    # Check for API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        exit(1)

    # Test with sample text
    sample_text = """
    Healthcare documentation is crucial for patient care. Medical records include patient history, 
    diagnosis information, treatment plans, and medication details. Healthcare providers must maintain 
    accurate documentation to ensure continuity of care and regulatory compliance.

    Electronic Health Records (EHR) systems help organize patient information digitally. These systems 
    store lab results, imaging studies, clinical notes, and billing information in a centralized location.
    """

    try:
        # Test with different models
        for embedding_model in ["all-MiniLM-L6-v2", "bge-small-en-v1.5"]:
            for llm_model in ["llama3-8b-8192", "llama3-70b-8192"]:
                print(f"\n🧪 Testing with {embedding_model} + {llm_model}")
                
                pipeline = create_rag_pipeline(
                    groq_api_key=groq_api_key,
                    embedding_model=embedding_model,
                    llm_model=llm_model
                )
                store_dir = pipeline.process_document(sample_text)

                # Test query
                result = pipeline.query_documents("What are the components of medical records?")
                print(f"Answer: {result['answer'][:100]}...")
                print(f"Config: {result['configuration']}")

    except Exception as e:
        print(f"Test failed: {e}")
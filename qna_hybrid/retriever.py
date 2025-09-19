
import os
import logging
from typing import Dict, Any, Optional
from qna_pipeline import HealthcareRAGPipeline, create_rag_pipeline

logger = logging.getLogger(__name__)

class HealthcareRetriever:
    """
    Enhanced retriever that works with the proper RAG pipeline
    """

    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.pipeline = None

    def get_query_engine(self, store_dir: str, **kwargs):
        """
        Get query engine for a specific store directory
        """
        try:
            if not self.pipeline:
                self.pipeline = create_rag_pipeline(self.groq_api_key)

            return self.pipeline.create_query_engine(store_dir, **kwargs)

        except Exception as e:
            logger.error(f"Error creating query engine: {e}")
            raise

    def query_documents(self, query: str, store_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Query documents in the specified store directory
        """
        try:
            if not self.pipeline:
                self.pipeline = create_rag_pipeline(self.groq_api_key)

            return self.pipeline.query_documents(query, store_dir, **kwargs)

        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise

# Legacy compatibility functions
def get_query_engine(store_dir: str, groq_api_key: str = None):
    """Legacy function for backward compatibility"""
    retriever = HealthcareRetriever(groq_api_key)
    return retriever.get_query_engine(store_dir)

def get_direct_query_engine(extracted_text: str, groq_api_key: str = None):
    """Create query engine directly from extracted text"""
    try:
        pipeline = create_rag_pipeline(groq_api_key or os.getenv("GROQ_API_KEY"))
        store_dir = pipeline.process_document(extracted_text)
        return pipeline.create_query_engine(store_dir)
    except Exception as e:
        logger.error(f"Error creating direct query engine: {e}")
        raise

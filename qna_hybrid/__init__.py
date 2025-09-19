"""
Healthcare RAG Q&A System Package
"""

# Import main classes and functions
try:
    from .qna_pipeline import HealthcareRAGPipeline, create_rag_pipeline
    # Import retriever if it exists
    try:
        from .retriever import HealthcareRetriever
    except ImportError:
        HealthcareRetriever = None
        
except ImportError as e:
    print(f"Warning: Could not import from qna_hybrid modules: {e}")
    HealthcareRAGPipeline = None
    create_rag_pipeline = None
    HealthcareRetriever = None

__version__ = "1.0.0"
__all__ = ['HealthcareRAGPipeline', 'create_rag_pipeline', 'HealthcareRetriever']

# Make sure the classes are available at package level
if HealthcareRAGPipeline is not None:
    __all__.append('HealthcareRAGPipeline')
if create_rag_pipeline is not None:
    __all__.append('create_rag_pipeline')
if HealthcareRetriever is not None:
    __all__.append('HealthcareRetriever')
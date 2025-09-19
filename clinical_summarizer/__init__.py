# clinical_summarizer/__init__.py

"""
Clinical Summarizer Package

This package provides clinical document summarization and medical entity extraction
capabilities for healthcare applications.

Components:
- summarizer: Clinical document summarization using LLM
- entity_extractor: Medical entity extraction from clinical texts
"""

from .summarizer import ClinicalSummarizer, create_clinical_summarizer
from .entity_extractor import MedicalEntityExtractor, create_entity_extractor

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"

__all__ = [
    "ClinicalSummarizer",
    "create_clinical_summarizer", 
    "MedicalEntityExtractor",
    "create_entity_extractor"
]
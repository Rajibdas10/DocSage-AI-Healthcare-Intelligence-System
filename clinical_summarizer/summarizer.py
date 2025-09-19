import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from llama_index.core import Document, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class ClinicalSummarizer:
    """
    Clinical document summarization system using Groq LLM
    Supports multiple summarization types for healthcare documents
    """
    
    def __init__(
        self,
        groq_api_key: str,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the clinical summarizer
        
        Args:
            groq_api_key: Groq API key
            llm_model: Groq LLM model to use
            embedding_model: Embedding model for document processing
        """
        self.groq_api_key = groq_api_key
        self.llm_model = llm_model
        
        # Initialize LLM
        self.llm = Groq(
            model=llm_model,
            api_key=groq_api_key,
            temperature=0.1  # Low temperature for consistent clinical summaries
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        logger.info(f"Clinical Summarizer initialized with {llm_model}")

    def _get_summary_prompt(self, summary_type: str, text: str) -> str:
        """
        Generate appropriate prompt based on summary type
        
        Args:
            summary_type: Type of summary to generate
            text: Clinical text to summarize
            
        Returns:
            Formatted prompt string
        """
        
        base_instruction = """You are a clinical AI assistant specialized in medical document summarization. 
        Provide accurate, professional, and well-structured summaries following medical documentation standards."""
        
        prompts = {
            "clinical_summary": f"""
            {base_instruction}
            
            Please provide a comprehensive clinical summary of the following medical document.
            
            Structure your summary with these sections:
            1. **Patient Overview**: Key demographics and presenting concerns
            2. **Clinical Findings**: Important examination findings, test results, diagnoses
            3. **Treatment Plan**: Medications, procedures, interventions
            4. **Follow-up**: Next steps, monitoring requirements, appointments
            5. **Key Considerations**: Critical alerts, contraindications, special notes
            
            Document to summarize:
            {text}
            
            Clinical Summary:
            """,
            
            "discharge_summary": f"""
            {base_instruction}
            
            Create a discharge summary from the following clinical information.
            
            Include these sections:
            1. **Admission Details**: Date, chief complaint, reason for admission
            2. **Hospital Course**: Key events, treatments provided, complications
            3. **Discharge Condition**: Current status, vital signs, functional capacity
            4. **Medications**: Discharge medications with instructions
            5. **Follow-up Care**: Appointments, monitoring, when to seek care
            6. **Patient Instructions**: Activity restrictions, diet, wound care
            
            Clinical Information:
            {text}
            
            Discharge Summary:
            """,
            
            "progress_note": f"""
            {base_instruction}
            
            Generate a progress note summary from the following clinical information.
            
            Use SOAP format:
            **Subjective**: Patient's reported symptoms, concerns, complaints
            **Objective**: Physical examination findings, vital signs, test results
            **Assessment**: Current diagnosis, clinical impression, problem list
            **Plan**: Treatment modifications, new orders, follow-up plans
            
            Clinical Information:
            {text}
            
            Progress Note:
            """,
            
            "diagnostic_summary": f"""
            {base_instruction}
            
            Provide a diagnostic summary focusing on clinical findings and conclusions.
            
            Include:
            1. **Primary Findings**: Key diagnostic results, abnormalities
            2. **Clinical Significance**: Medical interpretation of findings
            3. **Differential Diagnosis**: Possible conditions considered
            4. **Recommendations**: Suggested next steps, additional testing
            5. **Urgency Level**: Priority and timeline for action
            
            Diagnostic Information:
            {text}
            
            Diagnostic Summary:
            """,
            
            "medication_summary": f"""
            {base_instruction}
            
            Create a medication summary from the clinical information provided.
            
            Structure:
            1. **Current Medications**: Active prescriptions with dosing
            2. **Recent Changes**: New medications, discontinued drugs, dose adjustments
            3. **Drug Interactions**: Potential interactions identified
            4. **Allergies/Contraindications**: Known drug allergies, contraindications
            5. **Monitoring Requirements**: Lab tests, vital signs to monitor
            6. **Patient Education**: Key instructions for medication management
            
            Clinical Information:
            {text}
            
            Medication Summary:
            """,
            
            "brief_summary": f"""
            {base_instruction}
            
            Provide a concise clinical summary highlighting the most important information.
            
            Keep it brief but comprehensive:
            - **Primary Concern**: Main medical issue
            - **Key Findings**: Most significant clinical findings
            - **Current Status**: Patient's current condition
            - **Next Steps**: Immediate priorities
            
            Clinical Information:
            {text}
            
            Brief Clinical Summary:
            """
        }
        
        return prompts.get(summary_type, prompts["clinical_summary"])

    def generate_summary(
        self,
        clinical_text: str,
        summary_type: str = "clinical_summary",
        max_length: Optional[int] = None,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate clinical summary from provided text
        
        Args:
            clinical_text: Clinical document text to summarize
            summary_type: Type of summary to generate
            max_length: Maximum length of summary (optional)
            custom_instructions: Additional instructions for summarization
            
        Returns:
            Dictionary containing summary and metadata
        """
        
        try:
            # Validate inputs
            if not clinical_text or not clinical_text.strip():
                raise ValueError("Clinical text cannot be empty")
            
            if len(clinical_text.strip()) < 50:
                raise ValueError("Clinical text too short for meaningful summarization")
            
            # Get appropriate prompt
            prompt = self._get_summary_prompt(summary_type, clinical_text)
            
            # Add custom instructions if provided
            if custom_instructions:
                prompt += f"\n\nAdditional Instructions: {custom_instructions}"
            
            # Add length constraint if specified
            if max_length:
                prompt += f"\n\nPlease keep the summary under {max_length} words."
            
            logger.info(f"Generating {summary_type} summary")
            
            # Generate summary using LLM
            messages = [
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            summary = response.message.content.strip()
            
            # Calculate metrics
            word_count = len(summary.split())
            char_count = len(summary)
            compression_ratio = len(clinical_text) / len(summary) if summary else 0
            
            result = {
                "summary": summary,
                "summary_type": summary_type,
                "metadata": {
                    "original_length": len(clinical_text),
                    "summary_length": len(summary),
                    "word_count": word_count,
                    "character_count": char_count,
                    "compression_ratio": round(compression_ratio, 2),
                    "model_used": self.llm_model,
                    "generated_at": datetime.now().isoformat(),
                    "processing_time": None  # Can be added with timing decorator
                }
            }
            
            logger.info(f"Summary generated successfully. Length: {word_count} words")
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")

    def batch_summarize(
        self,
        documents: List[str],
        summary_type: str = "clinical_summary",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple documents
        
        Args:
            documents: List of clinical texts to summarize
            summary_type: Type of summary to generate
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of summary results
        """
        
        results = []
        total_docs = len(documents)
        
        for i, doc_text in enumerate(documents):
            try:
                # Generate summary
                summary_result = self.generate_summary(doc_text, summary_type)
                summary_result["batch_index"] = i
                results.append(summary_result)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, total_docs)
                    
            except Exception as e:
                logger.error(f"Error summarizing document {i}: {str(e)}")
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "summary": None
                })
        
        return results

    def get_summary_types(self) -> List[Dict[str, str]]:
        """
        Get available summary types with descriptions
        
        Returns:
            List of summary types and descriptions
        """
        
        return [
            {
                "type": "clinical_summary",
                "name": "Clinical Summary",
                "description": "Comprehensive overview with patient info, findings, treatment, and follow-up"
            },
            {
                "type": "discharge_summary",
                "name": "Discharge Summary",
                "description": "Summary for patient discharge including medications and follow-up care"
            },
            {
                "type": "progress_note",
                "name": "Progress Note",
                "description": "SOAP format progress note with subjective, objective, assessment, and plan"
            },
            {
                "type": "diagnostic_summary",
                "name": "Diagnostic Summary",
                "description": "Focus on diagnostic findings, interpretations, and recommendations"
            },
            {
                "type": "medication_summary",
                "name": "Medication Summary",
                "description": "Comprehensive medication review with interactions and monitoring"
            },
            {
                "type": "brief_summary",
                "name": "Brief Summary",
                "description": "Concise overview of key clinical information"
            }
        ]

def create_clinical_summarizer(
    groq_api_key: str,
    llm_model: str = "llama-3.3-70b-versatile",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> ClinicalSummarizer:
    """
    Factory function to create a clinical summarizer instance
    
    Args:
        groq_api_key: Groq API key
        llm_model: Groq LLM model to use
        embedding_model: Embedding model for document processing
        
    Returns:
        ClinicalSummarizer instance
    """
    
    return ClinicalSummarizer(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        embedding_model=embedding_model
    )
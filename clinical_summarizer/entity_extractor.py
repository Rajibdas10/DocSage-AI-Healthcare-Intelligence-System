import re
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq

logger = logging.getLogger(__name__)

class MedicalEntityExtractor:
    """
    Medical entity extraction from clinical documents using LLM-based extraction
    Identifies key medical entities like medications, conditions, procedures, etc.
    """
    
    def __init__(self, groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the medical entity extractor
        
        Args:
            groq_api_key: Groq API key
            llm_model: Groq LLM model to use
        """
        self.groq_api_key = groq_api_key
        self.llm_model = llm_model
        
        # Initialize LLM with specific settings for entity extraction
        self.llm = Groq(
            model=llm_model,
            api_key=groq_api_key,
            temperature=0.0  # Very low temperature for consistent extraction
        )
        
        logger.info(f"Medical Entity Extractor initialized with {llm_model}")
    
    def _get_extraction_prompt(self, text: str) -> str:
        """
        Generate prompt for medical entity extraction
        
        Args:
            text: Clinical text for entity extraction
            
        Returns:
            Formatted extraction prompt
        """
        
        prompt = f"""
        You are a medical AI assistant specialized in extracting key medical entities from clinical documents.
        
        Extract the following medical entities from the provided clinical text. Return the results in JSON format.
        
        Entity Categories to Extract:
        1. **medications**: Drug names, dosages, frequencies, routes
        2. **conditions**: Diagnoses, symptoms, medical conditions
        3. **procedures**: Medical procedures, surgeries, treatments performed
        4. **vital_signs**: Blood pressure, heart rate, temperature, respiratory rate, oxygen saturation
        5. **lab_results**: Laboratory test names and values
        6. **allergies**: Known allergies and adverse reactions
        7. **body_parts**: Anatomical locations mentioned
        8. **symptoms**: Patient-reported symptoms
        9. **medical_devices**: Implants, prosthetics, monitoring devices
        10. **healthcare_providers**: Doctors, nurses, specialists mentioned
        
        Format your response as JSON with this structure:
        ```json
        {{
            "medications": [
                {{
                    "name": "medication_name",
                    "dosage": "dosage_info",
                    "frequency": "frequency_info",
                    "route": "administration_route"
                }}
            ],
            "conditions": [
                {{
                    "name": "condition_name",
                    "status": "active/resolved/suspected",
                    "severity": "mild/moderate/severe"
                }}
            ],
            "procedures": [
                {{
                    "name": "procedure_name",
                    "date": "date_if_mentioned",
                    "location": "body_location"
                }}
            ],
            "vital_signs": [
                {{
                    "type": "vital_sign_type",
                    "value": "measurement_value",
                    "unit": "measurement_unit",
                    "date": "date_if_mentioned"
                }}
            ],
            "lab_results": [
                {{
                    "test_name": "test_name",
                    "value": "test_value",
                    "unit": "unit",
                    "reference_range": "normal_range",
                    "status": "normal/abnormal/critical"
                }}
            ],
            "allergies": [
                {{
                    "allergen": "allergen_name",
                    "reaction": "reaction_type",
                    "severity": "mild/moderate/severe"
                }}
            ],
            "body_parts": ["list", "of", "body_parts"],
            "symptoms": ["list", "of", "symptoms"],
            "medical_devices": ["list", "of", "devices"],
            "healthcare_providers": [
                {{
                    "name": "provider_name",
                    "specialty": "medical_specialty"
                }}
            ]
        }}
        ```
        
        Clinical Text:
        {text}
        
        Extract all relevant medical entities and return only the JSON response:
        """
        
        return prompt
    
    def extract_entities(self, clinical_text: str) -> Dict[str, Any]:
        """
        Extract medical entities from clinical text
        
        Args:
            clinical_text: Clinical document text
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        
        try:
            if not clinical_text or not clinical_text.strip():
                raise ValueError("Clinical text cannot be empty")
            
            logger.info("Starting medical entity extraction")
            
            # Generate extraction prompt
            prompt = self._get_extraction_prompt(clinical_text)
            
            # Extract entities using LLM
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = self.llm.chat(messages)
            
            # Parse JSON response
            response_text = response.message.content.strip()
            
            # Clean response to extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            try:
                entities = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Try to extract entities using regex fallback
                entities = self._fallback_extraction(clinical_text)
            
            # Add metadata
            result = {
                "entities": entities,
                "metadata": {
                    "total_entities": self._count_entities(entities),
                    "extraction_method": "llm_based",
                    "model_used": self.llm_model,
                    "extracted_at": datetime.now().isoformat(),
                    "text_length": len(clinical_text)
                }
            }
            
            logger.info(f"Entity extraction completed. Found {result['metadata']['total_entities']} entities")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise Exception(f"Entity extraction failed: {str(e)}")
    
    def _count_entities(self, entities: Dict[str, Any]) -> int:
        """Count total number of entities extracted"""
        total = 0
        for key, value in entities.items():
            if isinstance(value, list):
                total += len(value)
        return total
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """
        Fallback entity extraction using regex patterns
        Used when JSON parsing fails
        """
        
        entities = {
            "medications": [],
            "conditions": [],
            "procedures": [],
            "vital_signs": [],
            "lab_results": [],
            "allergies": [],
            "body_parts": [],
            "symptoms": [],
            "medical_devices": [],
            "healthcare_providers": []
        }
        
        # Basic regex patterns for common medical entities
        medication_patterns = [
            r'\b(?:mg|mcg|g|ml|units?)\b',
            r'\b\d+\s*(?:mg|mcg|g|ml|units?)\b',
            r'\b(?:twice daily|once daily|every \d+ hours|PRN|as needed)\b'
        ]
        
        vital_patterns = [
            r'\b(?:BP|blood pressure)\s*:?\s*\d+\/\d+',
            r'\bHR\s*:?\s*\d+',
            r'\btemp(?:erature)?\s*:?\s*\d+\.?\d*',
            r'\bO2\s*sat\s*:?\s*\d+%'
        ]
        
        # Extract using patterns (simplified implementation)
        text_lower = text.lower()
        
        # This is a basic fallback - in production you'd want more sophisticated patterns
        for pattern in medication_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                entities["medications"].append({"name": match, "dosage": "", "frequency": "", "route": ""})
        
        return entities
    
    def extract_specific_entity_type(
        self,
        clinical_text: str,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """
        Extract specific type of medical entity
        
        Args:
            clinical_text: Clinical text
            entity_type: Type of entity to extract (medications, conditions, etc.)
            
        Returns:
            List of extracted entities of specified type
        """
        
        all_entities = self.extract_entities(clinical_text)
        return all_entities["entities"].get(entity_type, [])
    
    def get_entity_summary(self, entities: Dict[str, Any]) -> Dict[str, int]:
        """
        Get summary statistics of extracted entities
        
        Args:
            entities: Extracted entities dictionary
            
        Returns:
            Summary statistics
        """
        
        summary = {}
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                summary[entity_type] = len(entity_list)
            else:
                summary[entity_type] = 1 if entity_list else 0
        
        return summary
    
    def filter_entities_by_confidence(
        self,
        entities: Dict[str, Any],
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Filter entities by confidence score (if available)
        
        Args:
            entities: Extracted entities
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Filtered entities
        """
        
        # Note: This is a placeholder for confidence-based filtering
        # In practice, you'd need to implement confidence scoring
        # For now, return all entities
        return entities

def create_entity_extractor(
    groq_api_key: str,
    llm_model: str = "llama-3.3-70b-versatile"
) -> MedicalEntityExtractor:
    """
    Factory function to create a medical entity extractor
    
    Args:
        groq_api_key: Groq API key
        llm_model: LLM model to use
        
    Returns:
        MedicalEntityExtractor instance
    """
    
    return MedicalEntityExtractor(
        groq_api_key=groq_api_key,
        llm_model=llm_model
    )
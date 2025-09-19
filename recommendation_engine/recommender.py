"""
Healthcare Recommendation Engine
Provides personalized recommendations for diet, yoga, medication, and lifestyle based on patient data.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from groq import Groq
from .rules import RecommendationRules

logger = logging.getLogger(__name__)


class HealthcareRecommender:
    """
    Healthcare recommendation engine that generates personalized recommendations
    for diet, yoga, medication management, and lifestyle based on patient data.
    """
    
    def __init__(self, groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Healthcare Recommender.
        
        Args:
            groq_api_key (str): Groq API key for LLM access
            llm_model (str): LLM model to use for generating recommendations
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model = llm_model
        self.rules = RecommendationRules()
        
    def generate_recommendations(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive healthcare recommendations based on patient data.
        
        Args:
            patient_data (dict): Patient information including demographics, conditions, preferences
            
        Returns:
            dict: Structured recommendations with categories and explanations
        """
        try:
            logger.info("Starting recommendation generation")
            
            # Apply rule-based recommendations first
            rule_based_recs = self._apply_rule_based_recommendations(patient_data)
            
            # Generate AI-enhanced recommendations
            ai_enhanced_recs = self._generate_ai_recommendations(patient_data, rule_based_recs)
            
            # Combine and format final recommendations
            final_recommendations = self._combine_recommendations(rule_based_recs, ai_enhanced_recs, patient_data)
            
            logger.info("Recommendations generated successfully")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise Exception(f"Failed to generate recommendations: {str(e)}")
    
    def _apply_rule_based_recommendations(self, patient_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Apply rule-based logic for basic recommendations."""
        
        recommendations = {
            "dietary": [],
            "exercise": [],
            "yoga": [],
            "medication": [],
            "lifestyle": []
        }
        
        age = patient_data.get('age', 30)
        gender = patient_data.get('gender', 'Unknown')
        conditions = patient_data.get('conditions', '').lower()
        activity_level = patient_data.get('activity_level', 'Moderate')
        dietary_restrictions = patient_data.get('dietary_restrictions', [])
        
        # Apply dietary rules
        dietary_recs = self.rules.get_dietary_recommendations(
            age, gender, conditions, dietary_restrictions
        )
        recommendations["dietary"].extend(dietary_recs)
        
        # Apply exercise rules
        exercise_recs = self.rules.get_exercise_recommendations(
            age, gender, conditions, activity_level
        )
        recommendations["exercise"].extend(exercise_recs)
        
        # Apply yoga recommendations
        yoga_recs = self.rules.get_yoga_recommendations(
            age, gender, conditions, activity_level
        )
        recommendations["yoga"].extend(yoga_recs)
        
        # Apply medication management rules
        medication_recs = self.rules.get_medication_recommendations(
            age, gender, conditions
        )
        recommendations["medication"].extend(medication_recs)
        
        # Apply lifestyle rules
        lifestyle_recs = self.rules.get_lifestyle_recommendations(
            age, gender, conditions
        )
        recommendations["lifestyle"].extend(lifestyle_recs)
        
        return recommendations
    


    def _generate_ai_recommendations(self, patient_data: Dict[str, Any], rule_based_recs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate AI-enhanced recommendations using LLM."""

        # Stronger JSON-only prompt
        prompt = f"""
        You are a healthcare recommendation specialist. 
        Generate enhanced, personalized, evidence-based recommendations. 
        Always prioritize patient safety and advise consulting healthcare professionals. 

        Output ONLY valid JSON with this structure:
        {{
            "enhanced_dietary": ["..."],
            "enhanced_exercise": ["..."],
            "enhanced_yoga": ["..."],
            "enhanced_medication": ["..."],
            "enhanced_lifestyle": ["..."],
            "personalized_notes": ["..."]
        }}

        Patient Data:
        Age: {patient_data.get("age")}
        Gender: {patient_data.get("gender")}
        Conditions: {patient_data.get("conditions")}
        Activity Level: {patient_data.get("activity_level")}
        Dietary Restrictions: {patient_data.get("dietary_restrictions")}
        
        Rule-Based Recommendations (basic):
        {json.dumps(rule_based_recs, indent=2)}
        """

        def safe_json_parse(text: str) -> dict:
            """Try parsing JSON safely, fallback to extracting JSON substring."""
            try:
                return json.loads(text)
            except:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except:
                        logger.warning("JSON substring found but parsing failed.")
                        return {}
                logger.warning("No valid JSON found in AI response.")
                return {}

        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a healthcare recommendation specialist. Respond ONLY in valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            ai_text = response.choices[0].message.content.strip()
            ai_recommendations = safe_json_parse(ai_text)

            # Always return all expected keys, even if AI misses some
            return {
                "enhanced_dietary": ai_recommendations.get("enhanced_dietary", []),
                "enhanced_exercise": ai_recommendations.get("enhanced_exercise", []),
                "enhanced_yoga": ai_recommendations.get("enhanced_yoga", []),
                "enhanced_medication": ai_recommendations.get("enhanced_medication", []),
                "enhanced_lifestyle": ai_recommendations.get("enhanced_lifestyle", []),
                "personalized_notes": ai_recommendations.get("personalized_notes", [])
            }

        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return {
                "enhanced_dietary": [],
                "enhanced_exercise": [],
                "enhanced_yoga": [],
                "enhanced_medication": [],
                "enhanced_lifestyle": [],
                "personalized_notes": []
            }

    
    def _create_recommendation_prompt(self, patient_data: Dict[str, Any], rule_based_recs: Dict[str, List[str]]) -> str:
        """Create a comprehensive prompt for AI recommendation generation."""
        
        prompt = f"""
        Based on the following patient information and initial rule-based recommendations, 
        provide enhanced, personalized healthcare recommendations:

        PATIENT INFORMATION:
        - Age: {patient_data.get('age', 'Unknown')}
        - Gender: {patient_data.get('gender', 'Unknown')}
        - Medical Conditions: {patient_data.get('conditions', 'None specified')}
        - Activity Level: {patient_data.get('activity_level', 'Moderate')}
        - Dietary Restrictions: {', '.join(patient_data.get('dietary_restrictions', [])) or 'None'}

        CURRENT RULE-BASED RECOMMENDATIONS:
        - Dietary: {', '.join(rule_based_recs.get('dietary', []))}
        - Exercise: {', '.join(rule_based_recs.get('exercise', []))}
        - Yoga: {', '.join(rule_based_recs.get('yoga', []))}
        - Medication: {', '.join(rule_based_recs.get('medication', []))}
        - Lifestyle: {', '.join(rule_based_recs.get('lifestyle', []))}

        Please provide enhanced recommendations in the following JSON format:
        {{
            "enhanced_dietary": ["specific dietary advice based on conditions"],
            "enhanced_exercise": ["tailored exercise recommendations"],
            "enhanced_yoga": ["specific yoga practices and poses"],
            "enhanced_medication": ["medication management tips"],
            "enhanced_lifestyle": ["lifestyle modification suggestions"],
            "personalized_notes": ["important notes specific to this patient"],
            "safety_warnings": ["any safety considerations"],
            "follow_up_recommendations": ["suggested follow-up actions"]
        }}

        Focus on:
        1. Evidence-based recommendations
        2. Patient-specific considerations
        3. Safety and contraindications
        4. Realistic and achievable goals
        5. Integration with existing treatments
        """
        
        return prompt
    
    def _combine_recommendations(self, rule_based: Dict[str, List[str]], 
                                ai_enhanced: Dict[str, Any], 
                                patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and AI recommendations into final output."""
        
        final_recommendations = {
            "patient_profile": {
                "age": patient_data.get('age'),
                "gender": patient_data.get('gender'),
                "conditions": patient_data.get('conditions'),
                "activity_level": patient_data.get('activity_level'),
                "dietary_restrictions": patient_data.get('dietary_restrictions', [])
            },
            "recommendations": {
                "dietary": {
                    "basic": rule_based.get('dietary', []),
                    "enhanced": ai_enhanced.get('enhanced_dietary', [])
                },
                "exercise": {
                    "basic": rule_based.get('exercise', []),
                    "enhanced": ai_enhanced.get('enhanced_exercise', [])
                },
                "yoga": {
                    "basic": rule_based.get('yoga', []),
                    "enhanced": ai_enhanced.get('enhanced_yoga', [])
                },
                "medication": {
                    "basic": rule_based.get('medication', []),
                    "enhanced": ai_enhanced.get('enhanced_medication', [])
                },
                "lifestyle": {
                    "basic": rule_based.get('lifestyle', []),
                    "enhanced": ai_enhanced.get('enhanced_lifestyle', [])
                }
            },
            "additional_insights": {
                "personalized_notes": ai_enhanced.get('personalized_notes', []),
                "safety_warnings": ai_enhanced.get('safety_warnings', []),
                "follow_up_recommendations": ai_enhanced.get('follow_up_recommendations', [])
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_used": self.llm_model,
                "recommendation_version": "1.0"
            }
        }
        
        return final_recommendations
    
    def analyze_patient_document(self, extracted_text: str) -> Dict[str, Any]:
        """
        Analyze patient document to extract relevant information for recommendations.
        
        Args:
            extracted_text (str): Text extracted from patient document
            
        Returns:
            dict: Extracted patient information
        """
        try:
            prompt = f"""
            Analyze the following medical document and extract relevant patient information 
            for generating healthcare recommendations. Focus on demographics, medical conditions, 
            medications, lifestyle factors, and any restrictions or preferences mentioned.

            DOCUMENT TEXT:
            {extracted_text}

            Please extract and return the information in the following JSON format:
            {{
                "demographics": {{
                    "age": "extracted age or null",
                    "gender": "extracted gender or null"
                }},
                "medical_conditions": ["list of medical conditions"],
                "current_medications": ["list of current medications"],
                "allergies": ["list of allergies"],
                "lifestyle_factors": {{
                    "activity_level": "extracted activity level or null",
                    "dietary_preferences": ["any dietary preferences or restrictions"],
                    "smoking_status": "smoker/non-smoker/former smoker or null",
                    "alcohol_consumption": "consumption level or null"
                }},
                "vital_signs": {{
                    "blood_pressure": "if mentioned",
                    "weight": "if mentioned",
                    "bmi": "if mentioned"
                }},
                "lab_results": ["any relevant lab results mentioned"],
                "symptoms": ["current symptoms mentioned"],
                "goals": ["any health goals or concerns mentioned"]
            }}

            If information is not available, use null or empty arrays as appropriate.
            """
            
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical document analyzer. Extract relevant information accurately and return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            extracted_info = json.loads(response.choices[0].message.content)
            logger.info("Patient document analyzed successfully")
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error analyzing patient document: {e}")
            raise Exception(f"Failed to analyze document: {str(e)}")


def create_recommender(groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile") -> HealthcareRecommender:
    """
    Factory function to create a HealthcareRecommender instance.
    
    Args:
        groq_api_key (str): Groq API key
        llm_model (str): LLM model name
        
    Returns:
        HealthcareRecommender: Configured recommender instance
    """
    return HealthcareRecommender(groq_api_key=groq_api_key, llm_model=llm_model)
"""
Healthcare Sentiment Analysis Classifier
Analyzes emotional tone and sentiment in healthcare communications, patient feedback, and clinical notes.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from groq import Groq

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Healthcare-focused sentiment analysis system that analyzes emotional tone,
    patient satisfaction, urgency levels, and specific healthcare emotions.
    """
    
    def __init__(self, groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Sentiment Analyzer.
        
        Args:
            groq_api_key (str): Groq API key for LLM access
            llm_model (str): LLM model to use for sentiment analysis
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model = llm_model
        self._initialize_healthcare_lexicon()
        
    def _initialize_healthcare_lexicon(self):
        """Initialize healthcare-specific sentiment lexicon."""
        
        # Healthcare-specific positive indicators
        self.positive_indicators = [
            "feeling better", "improved", "relieved", "comfortable", "satisfied",
            "grateful", "thankful", "healing", "recovery", "progress", "better",
            "excellent care", "professional", "caring", "helpful", "supportive"
        ]
        
        # Healthcare-specific negative indicators
        self.negative_indicators = [
            "pain", "worse", "concerned", "worried", "anxious", "frustrated",
            "uncomfortable", "dissatisfied", "confused", "scared", "afraid",
            "terrible", "awful", "unbearable", "suffering", "distressed"
        ]
        
        # Urgency indicators
        self.urgency_indicators = {
            "high": ["emergency", "urgent", "immediately", "severe", "critical", "acute", "intense"],
            "medium": ["concerned", "worried", "uncomfortable", "moderate", "significant"],
            "low": ["mild", "slight", "minor", "manageable", "routine"]
        }
        
        # Healthcare-specific emotions
        self.healthcare_emotions = {
            "anxiety": ["anxious", "worried", "nervous", "scared", "fearful", "panic"],
            "relief": ["relieved", "better", "comfortable", "ease", "calm"],
            "frustration": ["frustrated", "annoyed", "upset", "disappointed"],
            "gratitude": ["grateful", "thankful", "appreciate", "blessed"],
            "confusion": ["confused", "unclear", "don't understand", "puzzled"],
            "hope": ["hopeful", "optimistic", "confident", "positive outlook"],
            "trust": ["trust", "confidence", "faith", "believe in"],
            "concern": ["concerned", "worried about", "wondering", "question"]
        }
    
    def analyze_sentiment(self, text: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on healthcare text.
        
        Args:
            text (str): Text to analyze
            analysis_types (List[str]): Types of analysis to perform
            
        Returns:
            dict: Comprehensive sentiment analysis results
        """
        try:
            logger.info("Starting sentiment analysis")
            
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Default analysis types
            if analysis_types is None:
                analysis_types = ["Overall Sentiment", "Emotion Detection", "Urgency Level", "Patient Satisfaction"]
            
            # Perform rule-based analysis first
            rule_based_results = self._rule_based_analysis(text)
            
            # Perform AI-enhanced analysis
            ai_results = self._ai_sentiment_analysis(text, analysis_types)
            
            # Combine results
            final_results = self._combine_sentiment_results(rule_based_results, ai_results, text)
            
            logger.info("Sentiment analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise Exception(f"Failed to analyze sentiment: {str(e)}")
    
    def _rule_based_analysis(self, text: str) -> Dict[str, Any]:
        """Perform rule-based sentiment analysis using healthcare lexicon."""
        
        text_lower = text.lower()
        
        # Count positive and negative indicators
        positive_count = sum(1 for indicator in self.positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in self.negative_indicators if indicator in text_lower)
        
        # Calculate basic sentiment score
        total_indicators = positive_count + negative_count
        if total_indicators > 0:
            sentiment_score = (positive_count - negative_count) / total_indicators
        else:
            sentiment_score = 0.0
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Analyze urgency
        urgency_level = self._analyze_urgency(text_lower)
        
        # Analyze healthcare emotions
        emotions = self._analyze_healthcare_emotions(text_lower)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "urgency_level": urgency_level,
            "emotions": emotions,
            "key_phrases": key_phrases,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def _analyze_urgency(self, text_lower: str) -> str:
        """Analyze urgency level in the text."""
        
        urgency_scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, indicators in self.urgency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    urgency_scores[level] += 1
        
        # Determine urgency level
        if urgency_scores["high"] > 0:
            return "High"
        elif urgency_scores["medium"] > 0:
            return "Medium"
        elif urgency_scores["low"] > 0:
            return "Low"
        else:
            return "Medium"  # Default
    
    def _analyze_healthcare_emotions(self, text_lower: str) -> Dict[str, float]:
        """Analyze healthcare-specific emotions in the text."""
        
        emotion_scores = {}
        total_emotion_words = 0
        
        for emotion, indicators in self.healthcare_emotions.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            emotion_scores[emotion] = count
            total_emotion_words += count
        
        # Normalize scores
        if total_emotion_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_emotion_words
        else:
            # Default distribution
            emotion_scores = {emotion: 0.0 for emotion in self.healthcare_emotions.keys()}
            emotion_scores["concern"] = 0.3  # Default mild concern
            emotion_scores["hope"] = 0.2  # Default mild hope
        
        return emotion_scores
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from the text."""
        
        # Simple key phrase extraction based on healthcare context
        key_phrases = []
        
        # Look for phrases around healthcare keywords
        healthcare_keywords = [
            "feeling", "pain", "medication", "treatment", "doctor", "nurse",
            "appointment", "symptoms", "recovery", "care", "hospital", "clinic"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                for keyword in healthcare_keywords:
                    if keyword.lower() in sentence.lower():
                        # Extract a meaningful phrase around the keyword
                        phrase = self._extract_phrase_around_keyword(sentence, keyword)
                        if phrase and len(phrase) > 5:
                            key_phrases.append(phrase)
                            break
        
        return list(set(key_phrases))[:5]  # Return unique phrases, max 5
    
    def _extract_phrase_around_keyword(self, sentence: str, keyword: str) -> str:
        """Extract a phrase around a specific keyword."""
        
        words = sentence.split()
        keyword_indices = [i for i, word in enumerate(words) if keyword.lower() in word.lower()]
        
        if keyword_indices:
            idx = keyword_indices[0]
            start = max(0, idx - 2)
            end = min(len(words), idx + 3)
            phrase = " ".join(words[start:end])
            return phrase.strip()
        
        return ""
    
    def _ai_sentiment_analysis(self, text: str, analysis_types: List[str]) -> Dict[str, Any]:
        """Perform AI-enhanced sentiment analysis using LLM."""
        
        prompt = self._create_sentiment_prompt(text, analysis_types)
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a healthcare sentiment analysis specialist. 
                        Analyze healthcare-related text for emotional tone, patient satisfaction, 
                        urgency, and specific healthcare emotions. Always respond in valid JSON format."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            ai_results = json.loads(response.choices[0].message.content)
            return ai_results
            
        except Exception as e:
            logger.error(f"Error in AI sentiment analysis: {e}")
            # Return basic structure if AI fails
            return {
                "overall_sentiment": {"label": "Neutral", "score": 0.0, "confidence": 0.5},
                "detailed_emotions": {},
                "patient_satisfaction": "Unknown",
                "urgency_assessment": "Medium",
                "clinical_insights": [],
                "recommendations": []
            }
    
    def _create_sentiment_prompt(self, text: str, analysis_types: List[str]) -> str:
        """Create a comprehensive prompt for AI sentiment analysis."""
        
        prompt = f"""
        Analyze the following healthcare-related text for sentiment and emotional indicators:

        TEXT TO ANALYZE:
        "{text}"

        Please provide analysis for the following aspects: {', '.join(analysis_types)}

        Return your analysis in the following JSON format:
        {{
            "overall_sentiment": {{
                "label": "Positive/Negative/Neutral",
                "score": 0.0,
                "confidence": 0.0
            }},
            "detailed_emotions": {{
                "joy": 0.0,
                "trust": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0,
                "anxiety": 0.0,
                "relief": 0.0,
                "frustration": 0.0,
                "gratitude": 0.0,
                "confusion": 0.0,
                "hope": 0.0,
                "concern": 0.0
            }},
            "patient_satisfaction": "Very Satisfied/Satisfied/Neutral/Dissatisfied/Very Dissatisfied",
            "urgency_assessment": "Low/Medium/High",
            "clinical_insights": [
                "List of clinical or emotional insights from the text"
            ],
            "key_emotional_indicators": [
                "List of specific words or phrases that indicate emotion"
            ],
            "recommendations": [
                "Suggestions for healthcare providers based on the sentiment"
            ],
            "context_analysis": {{
                "is_patient_feedback": true/false,
                "is_clinical_note": true/false,
                "communication_type": "complaint/inquiry/appreciation/concern/other",
                "requires_immediate_attention": true/false
            }}
        }}

        Focus on:
        1. Healthcare-specific emotions and concerns
        2. Patient experience indicators
        3. Communication urgency and priority
        4. Clinical relevance of emotional state
        5. Actionable insights for healthcare providers
        """
        
        return prompt
    
    def _combine_sentiment_results(self, rule_based: Dict[str, Any], 
                                  ai_results: Dict[str, Any], 
                                  original_text: str) -> Dict[str, Any]:
        """Combine rule-based and AI sentiment analysis results."""
        
        # Combine emotion scores (average of rule-based and AI)
        combined_emotions = {}
        rule_emotions = rule_based.get("emotions", {})
        ai_emotions = ai_results.get("detailed_emotions", {})
        
        # Get all unique emotion keys
        all_emotions = set(list(rule_emotions.keys()) + list(ai_emotions.keys()))
        
        for emotion in all_emotions:
            rule_score = rule_emotions.get(emotion, 0.0)
            ai_score = ai_emotions.get(emotion, 0.0)
            combined_emotions[emotion] = (rule_score + ai_score) / 2
        
        # Determine final sentiment (weighted combination)
        ai_sentiment = ai_results.get("overall_sentiment", {})
        rule_sentiment_score = rule_based.get("sentiment_score", 0.0)
        ai_sentiment_score = ai_sentiment.get("score", 0.0)
        
        # Weight AI results more heavily but consider rule-based
        final_sentiment_score = (ai_sentiment_score * 0.7) + (rule_sentiment_score * 0.3)
        
        # Determine final label
        if final_sentiment_score > 0.1:
            final_label = "Positive"
        elif final_sentiment_score < -0.1:
            final_label = "Negative"
        else:
            final_label = "Neutral"
        
        # Combine urgency assessment
        rule_urgency = rule_based.get("urgency_level", "Medium")
        ai_urgency = ai_results.get("urgency_assessment", "Medium")
        
        # Use AI urgency if it suggests higher urgency, otherwise use rule-based
        urgency_priority = {"Low": 1, "Medium": 2, "High": 3}
        final_urgency = rule_urgency
        if urgency_priority.get(ai_urgency, 2) > urgency_priority.get(rule_urgency, 2):
            final_urgency = ai_urgency
        
        # Create final combined results
        final_results = {
            "overall_sentiment": {
                "label": final_label,
                "score": final_sentiment_score,
                "confidence": ai_sentiment.get("confidence", 0.7)
            },
            "emotions": combined_emotions,
            "patient_satisfaction": ai_results.get("patient_satisfaction", "Unknown"),
            "urgency_level": final_urgency,
            "key_phrases": rule_based.get("key_phrases", []),
            "clinical_insights": ai_results.get("clinical_insights", []),
            "emotional_indicators": ai_results.get("key_emotional_indicators", []),
            "recommendations": ai_results.get("recommendations", []),
            "context_analysis": ai_results.get("context_analysis", {}),
            "analysis_metadata": {
                "text_length": len(original_text),
                "analyzed_at": datetime.now().isoformat(),
                "model_used": self.llm_model,
                "analysis_version": "1.0",
                "methods_used": ["rule_based", "ai_enhanced"]
            },
            "detailed_scores": {
                "rule_based_sentiment": rule_based.get("sentiment_score", 0.0),
                "ai_sentiment": ai_sentiment_score,
                "positive_indicators": rule_based.get("positive_indicators", 0),
                "negative_indicators": rule_based.get("negative_indicators", 0)
            }
        }
        
        return final_results
    
    def analyze_batch_texts(self, texts: List[str], analysis_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to analyze
            analysis_types (List[str]): Types of analysis to perform
            
        Returns:
            List[Dict[str, Any]]: List of sentiment analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text, analysis_types)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "overall_sentiment": {"label": "Unknown", "score": 0.0, "confidence": 0.0}
                })
        
        return results
    
    def get_sentiment_summary(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get a summary of sentiment analysis for multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            Dict[str, Any]: Summary of sentiment analysis
        """
        batch_results = self.analyze_batch_texts(texts)
        
        # Calculate summary statistics
        sentiments = [r["overall_sentiment"]["label"] for r in batch_results if "error" not in r]
        sentiment_counts = {
            "Positive": sentiments.count("Positive"),
            "Negative": sentiments.count("Negative"),
            "Neutral": sentiments.count("Neutral")
        }
        
        # Average satisfaction
        satisfactions = [r.get("patient_satisfaction") for r in batch_results if "error" not in r]
        satisfaction_counts = {}
        for sat in satisfactions:
            if sat and sat != "Unknown":
                satisfaction_counts[sat] = satisfaction_counts.get(sat, 0) + 1
        
        # Urgency levels
        urgencies = [r.get("urgency_level") for r in batch_results if "error" not in r]
        urgency_counts = {
            "High": urgencies.count("High"),
            "Medium": urgencies.count("Medium"),
            "Low": urgencies.count("Low")
        }
        
        return {
            "total_texts": len(texts),
            "successful_analyses": len([r for r in batch_results if "error" not in r]),
            "sentiment_distribution": sentiment_counts,
            "satisfaction_distribution": satisfaction_counts,
            "urgency_distribution": urgency_counts,
            "requires_attention": len([r for r in batch_results if r.get("urgency_level") == "High"]),
            "generated_at": datetime.now().isoformat()
        }


def create_sentiment_analyzer(groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile") -> SentimentAnalyzer:
    """
    Factory function to create a SentimentAnalyzer instance.
    
    Args:
        groq_api_key (str): Groq API key
        llm_model (str): LLM model name
        
    Returns:
        SentimentAnalyzer: Configured sentiment analyzer instance
    """
    return SentimentAnalyzer(groq_api_key=groq_api_key, llm_model=llm_model)
import os
from typing import Dict, Any
from .sentiment_classifier import SentimentAnalyzer
from groq import Groq

class SentimentChatbot:
    """
    Chatbot that adapts responses using sentiment analysis + LLM.
    """
    
    def __init__(self, groq_api_key: str, llm_model: str = "llama-3.3-70b-versatile"):
        self.analyzer = SentimentAnalyzer(groq_api_key=groq_api_key, llm_model=llm_model)
        self.llm_model = llm_model
        self.groq_api_key = groq_api_key
    
    def generate_response(self, user_input: str, chat_history: list) -> Dict[str, Any]:
        # Step 1: Run sentiment analysis
        sentiment_results = self.analyzer.analyze_sentiment(user_input)
        
        sentiment_label = sentiment_results["overall_sentiment"]["label"]
        emotions = sentiment_results.get("emotions", {})
        urgency = sentiment_results.get("urgency_level", "Medium")
        
        # Step 2: Build prompt with context
        system_prompt = f"""
        You are a supportive, empathetic healthcare assistant.
        Sentiment detected: {sentiment_label}
        Emotions: {emotions}
        Urgency: {urgency}
        
        - If negative + high sadness/anxiety → be more comforting.
        - If frustration → acknowledge and calm.
        - If positive → encourage and reinforce positivity.
        - If urgency is High → strongly recommend professional help.
        
        Keep responses short, natural, and conversational (like WhatsApp/ChatGPT).
        """
        
        # Step 3: Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)  # maintain previous conversation
        messages.append({"role": "user", "content": user_input})
        
        client = Groq(api_key=self.groq_api_key)
        
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        bot_reply = response.choices[0].message.content 
        
        return {
            "user_input": user_input,
            "analysis": sentiment_results,
            "chatbot_response": bot_reply
        }
"""
Recommendation Engine Package
Provides personalized healthcare recommendations for diet, yoga, medication, and lifestyle.
"""

from .recommender import HealthcareRecommender, create_recommender
from .rules import RecommendationRules

__all__ = ['HealthcareRecommender', 'create_recommender', 'RecommendationRules']
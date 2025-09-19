"""
Rule-based Healthcare Recommendation System
Contains predefined rules and logic for generating basic healthcare recommendations.
"""

from typing import List, Dict, Any


class RecommendationRules:
    """
    Rule-based recommendation system for healthcare advice.
    Contains predefined logic for diet, yoga, medication, and lifestyle recommendations.
    """
    
    def __init__(self):
        """Initialize the rule-based recommendation system."""
        self._initialize_rule_database()
    
    def _initialize_rule_database(self):
        """Initialize the rule database with predefined healthcare rules."""
        
        # Age-based dietary recommendations
        self.age_dietary_rules = {
            (0, 18): ["Focus on growth-supporting nutrients", "Include calcium-rich foods", "Limit processed foods"],
            (19, 30): ["Maintain balanced macronutrients", "Include antioxidant-rich foods", "Stay hydrated"],
            (31, 50): ["Focus on heart-healthy foods", "Include fiber-rich options", "Monitor sodium intake"],
            (51, 65): ["Include bone-supporting nutrients", "Focus on lean proteins", "Limit saturated fats"],
            (66, 120): ["Soft, easily digestible foods", "High-protein options", "Adequate hydration"]
        }
        
        # Condition-specific dietary rules
        self.condition_dietary_rules = {
            "diabetes": ["Low glycemic index foods", "Regular meal timing", "Monitor carbohydrate intake", "Include fiber-rich foods"],
            "hypertension": ["Reduce sodium intake (<2300mg/day)", "Include potassium-rich foods", "DASH diet principles", "Limit processed foods"],
            "heart disease": ["Mediterranean diet", "Omega-3 rich foods", "Limit trans fats", "Include whole grains"],
            "obesity": ["Calorie-controlled diet", "Portion control", "Include lean proteins", "Increase vegetable intake"],
            "arthritis": ["Anti-inflammatory foods", "Omega-3 fatty acids", "Limit processed sugars", "Include turmeric and ginger"]
        }
        
        # Activity level exercise rules
        self.exercise_rules = {
            "sedentary": ["Start with 10-15 minutes daily", "Low-impact activities", "Walking program", "Gradual progression"],
            "light": ["20-30 minutes moderate exercise", "Include strength training 2x/week", "Flexibility exercises", "Outdoor activities"],
            "moderate": ["30-45 minutes regular exercise", "Mix cardio and strength training", "Sport activities", "Cross-training"],
            "active": ["45-60 minutes daily", "High-intensity intervals", "Competitive sports", "Advanced strength training"],
            "very active": ["60+ minutes daily", "Performance optimization", "Recovery protocols", "Professional guidance"]
        }
        
        # Condition-specific exercise rules
        self.condition_exercise_rules = {
            "diabetes": ["Regular aerobic exercise", "Post-meal walking", "Monitor blood sugar", "Resistance training 2-3x/week"],
            "hypertension": ["Moderate cardio 30 min/day", "Avoid heavy lifting", "Swimming or cycling", "Stress-reduction activities"],
            "heart disease": ["Supervised cardiac rehab", "Low to moderate intensity", "Avoid sudden exertion", "Regular monitoring"],
            "arthritis": ["Low-impact exercises", "Swimming or water aerobics", "Range of motion exercises", "Avoid high-impact activities"],
            "osteoporosis": ["Weight-bearing exercises", "Balance training", "Resistance exercises", "Fall prevention focus"]
        }
        
        # Yoga recommendations by condition and age
        self.yoga_rules = {
            "beginners": ["Hatha yoga", "Basic breathing exercises", "Simple poses", "15-20 minute sessions"],
            "diabetes": ["Restorative yoga", "Pranayama breathing", "Gentle twists", "Avoid inversions if complications"],
            "hypertension": ["Gentle yoga", "Avoid inversions", "Focus on relaxation", "Deep breathing exercises"],
            "arthritis": ["Chair yoga modifications", "Gentle stretches", "Warm-up important", "Avoid extreme positions"],
            "stress": ["Yin yoga", "Meditation practices", "Breathing techniques", "Restorative poses"]
        }
        
        # Medication management rules
        self.medication_rules = {
            "general": ["Take as prescribed", "Set consistent timing", "Use pill organizers", "Regular pharmacy reviews"],
            "diabetes": ["Monitor blood sugar regularly", "Time with meals appropriately", "Rotate injection sites", "Emergency supplies"],
            "hypertension": ["Daily monitoring", "Same time each day", "Don't skip doses", "Lifestyle modifications"],
            "heart disease": ["Carry emergency medications", "Regular cardiology follow-ups", "Monitor side effects", "Drug interaction awareness"]
        }
        
        # Lifestyle recommendations
        self.lifestyle_rules = {
            "sleep": ["7-9 hours nightly", "Consistent sleep schedule", "Sleep hygiene practices", "Dark, cool environment"],
            "stress": ["Relaxation techniques", "Regular exercise", "Social support", "Professional help if needed"],
            "smoking": ["Cessation programs", "Nicotine replacement options", "Support groups", "Healthcare provider guidance"],
            "alcohol": ["Moderate consumption limits", "Avoid with medications", "Monitor interactions", "Alternative beverages"]
        }
    
    def get_dietary_recommendations(self, age: int, gender: str, conditions: str, restrictions: List[str]) -> List[str]:
        """
        Get dietary recommendations based on patient parameters.
        
        Args:
            age (int): Patient age
            gender (str): Patient gender
            conditions (str): Medical conditions (comma-separated string)
            restrictions (List[str]): Dietary restrictions
            
        Returns:
            List[str]: Dietary recommendations
        """
        recommendations = []
        
        # Age-based recommendations
        for age_range, recs in self.age_dietary_rules.items():
            if age_range[0] <= age <= age_range[1]:
                recommendations.extend(recs)
                break
        
        # Condition-based recommendations
        conditions_lower = conditions.lower() if conditions else ""
        for condition, recs in self.condition_dietary_rules.items():
            if condition in conditions_lower:
                recommendations.extend(recs)
        
        # Handle dietary restrictions
        if "vegetarian" in [r.lower() for r in restrictions]:
            recommendations.append("Focus on plant-based proteins (legumes, nuts, seeds)")
            recommendations.append("Ensure adequate B12 and iron intake")
        
        if "vegan" in [r.lower() for r in restrictions]:
            recommendations.append("Include fortified foods for B12, D3")
            recommendations.append("Combine proteins for complete amino acids")
        
        if "gluten-free" in [r.lower() for r in restrictions]:
            recommendations.append("Choose naturally gluten-free grains (quinoa, rice)")
            recommendations.append("Check labels for hidden gluten")
        
        # Gender-specific recommendations
        if gender.lower() == "female":
            recommendations.append("Include iron-rich foods")
            recommendations.append("Ensure adequate calcium intake")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_exercise_recommendations(self, age: int, gender: str, conditions: str, activity_level: str) -> List[str]:
        """Get exercise recommendations based on patient parameters."""
        recommendations = []
        
        # Activity level recommendations
        activity_key = activity_level.lower() if activity_level else "moderate"
        if activity_key in self.exercise_rules:
            recommendations.extend(self.exercise_rules[activity_key])
        
        # Condition-based modifications
        conditions_lower = conditions.lower() if conditions else ""
        for condition, recs in self.condition_exercise_rules.items():
            if condition in conditions_lower:
                recommendations.extend(recs)
        
        # Age-specific modifications
        if age >= 65:
            recommendations.extend([
                "Include balance training",
                "Fall prevention focus",
                "Warm-up and cool-down important",
                "Consider supervised programs"
            ])
        elif age <= 18:
            recommendations.extend([
                "Include fun, varied activities",
                "Sports participation encouraged",
                "Proper form instruction important",
                "Growth plate considerations"
            ])
        
        return list(set(recommendations))
    
    def get_yoga_recommendations(self, age: int, gender: str, conditions: str, activity_level: str) -> List[str]:
        """Get yoga recommendations based on patient parameters."""
        recommendations = []
        
        # Basic recommendations for beginners or general population
        if activity_level.lower() in ["sedentary", "light"]:
            recommendations.extend(self.yoga_rules["beginners"])
        
        # Condition-specific yoga modifications
        conditions_lower = conditions.lower() if conditions else ""
        for condition in ["diabetes", "hypertension", "arthritis"]:
            if condition in conditions_lower:
                recommendations.extend(self.yoga_rules.get(condition, []))
        
        # Stress-related recommendations
        if any(word in conditions_lower for word in ["anxiety", "stress", "depression"]):
            recommendations.extend(self.yoga_rules["stress"])
        
        # Age-specific modifications
        if age >= 65:
            recommendations.extend([
                "Chair yoga modifications available",
                "Focus on gentle movements",
                "Balance and stability poses",
                "Shorter session durations"
            ])
        
        if not recommendations:  # Default recommendations
            recommendations = [
                "Start with basic Hatha yoga",
                "Focus on proper breathing",
                "15-30 minute sessions",
                "Listen to your body"
            ]
        
        return list(set(recommendations))
    
    def get_medication_recommendations(self, age: int, gender: str, conditions: str) -> List[str]:
        """Get medication management recommendations."""
        recommendations = []
        
        # General medication management
        recommendations.extend(self.medication_rules["general"])
        
        # Condition-specific medication rules
        conditions_lower = conditions.lower() if conditions else ""
        for condition, recs in self.medication_rules.items():
            if condition != "general" and condition in conditions_lower:
                recommendations.extend(recs)
        
        # Age-specific considerations
        if age >= 65:
            recommendations.extend([
                "Regular medication reviews",
                "Monitor for interactions",
                "Consider pill organizers",
                "Involve family in management"
            ])
        
        return list(set(recommendations))
    
    def get_lifestyle_recommendations(self, age: int, gender: str, conditions: str) -> List[str]:
        """Get lifestyle recommendations."""
        recommendations = []
        
        # Basic lifestyle recommendations
        recommendations.extend(self.lifestyle_rules["sleep"])
        recommendations.extend(self.lifestyle_rules["stress"])
        
        # Condition-specific lifestyle modifications
        conditions_lower = conditions.lower() if conditions else ""
        
        if any(word in conditions_lower for word in ["diabetes", "heart", "hypertension"]):
            recommendations.extend([
                "Regular health monitoring",
                "Maintain healthy weight",
                "Limit alcohol consumption",
                "Quit smoking if applicable"
            ])
        
        if "arthritis" in conditions_lower:
            recommendations.extend([
                "Joint protection strategies",
                "Heat/cold therapy",
                "Maintain healthy weight",
                "Ergonomic considerations"
            ])
        
        # Age-specific lifestyle recommendations
        if age >= 65:
            recommendations.extend([
                "Social engagement important",
                "Fall prevention measures",
                "Regular health screenings",
                "Emergency contact systems"
            ])
        elif age <= 30:
            recommendations.extend([
                "Establish healthy habits early",
                "Regular preventive care",
                "Stress management skills",
                "Work-life balance"
            ])
        
        return list(set(recommendations))
    
    def get_safety_considerations(self, age: int, conditions: str) -> List[str]:
        """Get safety considerations based on patient profile."""
        safety_warnings = []
        
        conditions_lower = conditions.lower() if conditions else ""
        
        # Age-related safety considerations
        if age >= 65:
            safety_warnings.extend([
                "Consult healthcare provider before starting new exercise",
                "Monitor for medication side effects",
                "Fall prevention measures important",
                "Regular health check-ups essential"
            ])
        
        # Condition-specific safety warnings
        if "heart" in conditions_lower:
            safety_warnings.extend([
                "Monitor heart rate during exercise",
                "Stop activity if chest pain occurs",
                "Carry emergency medications",
                "Avoid sudden intense activities"
            ])
        
        if "diabetes" in conditions_lower:
            safety_warnings.extend([
                "Monitor blood sugar before/after exercise",
                "Carry glucose tablets",
                "Check feet daily",
                "Stay hydrated"
            ])
        
        if "hypertension" in conditions_lower:
            safety_warnings.extend([
                "Monitor blood pressure regularly",
                "Avoid sudden position changes",
                "Limit sodium intake strictly",
                "Take medications as prescribed"
            ])
        
        return safety_warnings
import logging
from models.emotion_model import EmotionDetector

logger = logging.getLogger(__name__)

class EmotionService:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        
    def load_model(self):
        """Load the emotion detection model"""
        try:
            self.emotion_detector.load_model()
            logger.info("Emotion service model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion service model: {e}")
            raise
    
    def detect_emotion(self, text: str) -> dict:
        """
        Detect emotion from text input
        Returns: {"emotion": str, "confidence": float, "all_scores": dict}
        """
        try:
            result = self.emotion_detector.predict_emotion(text)
            logger.info(f"Emotion detected: {result['emotion']} ({result['confidence']:.2f})")
            return result
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_scores": {}
            }
    
    def get_emotion_context(self, emotion_result: dict) -> dict:
        """
        Get additional context for the detected emotion
        Returns: {"primary_emotion": str, "secondary_emotions": list, "intensity": str}
        """
        emotion = emotion_result.get("emotion", "neutral")
        confidence = emotion_result.get("confidence", 0.5)
        all_scores = emotion_result.get("all_scores", {})
        
        # Determine intensity based on confidence
        if confidence > 0.8:
            intensity = "high"
        elif confidence > 0.6:
            intensity = "medium"
        else:
            intensity = "low"
        
        # Get secondary emotions (emotions with >20% confidence)
        secondary_emotions = [
            (emotion_name, score) 
            for emotion_name, score in all_scores.items() 
            if score > 0.2 and emotion_name != emotion
        ]
        
        return {
            "primary_emotion": emotion,
            "secondary_emotions": secondary_emotions,
            "intensity": intensity,
            "confidence": confidence
        } 
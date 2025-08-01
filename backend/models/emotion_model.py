import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch  # ← ADD THIS IMPORT
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        self.model_name = "mananshah296/roberta-emotion"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        
    def load_model(self):
        """Load the RoBERTa emotion classification model"""
        try:
            logger.info(f"Loading emotion model on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            raise
    
    def predict_emotion(self, text: str) -> dict:
        """
        Predict emotion from text
        Returns: {"emotion": str, "confidence": float, "all_scores": dict, "secondary_emotions": dict}
        """
        if not self.model or not self.tokenizer:
            self.load_model()
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get results
            predictions = probabilities.cpu().numpy()[0]
            emotion_scores = {label: float(score) for label, score in zip(self.emotion_labels, predictions)}
            
            # Get top emotion
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[top_emotion]
            
            # Get secondary emotions (above 10% probability)
            secondary_emotions = {
                emotion: score for emotion, score in emotion_scores.items() 
                if score > 0.1 and emotion != top_emotion
            }
            
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "all_scores": emotion_scores,
                "secondary_emotions": secondary_emotions
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_scores": {},
                "secondary_emotions": {}
            }
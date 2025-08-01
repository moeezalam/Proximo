import logging
from models.llm_model import MistralConversation

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self):
        self.conversation_model = MistralConversation()
        
    def load_model(self):
        """Load the conversation model"""
        try:
            self.conversation_model.load_model()
            logger.info("Conversation service model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading conversation service model: {e}")
            raise
    
    def generate_response(self, user_input: str, emotion: str, confidence: float) -> str:
        """
        Generate an emotion-aware response to user input
        Returns: str - The generated response
        """
        try:
            response = self.conversation_model.generate_response(user_input, emotion, confidence)
            logger.info(f"Response generated: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
    
    def get_conversation_context(self, user_input: str, emotion_result: dict) -> dict:
        """
        Get conversation context for enhanced response generation
        Returns: dict with conversation context
        """
        emotion = emotion_result.get("emotion", "neutral")
        confidence = emotion_result.get("confidence", 0.5)
        
        # Analyze input characteristics
        input_length = len(user_input)
        has_question = "?" in user_input
        has_exclamation = "!" in user_input
        
        context = {
            "input_length": input_length,
            "has_question": has_question,
            "has_exclamation": has_exclamation,
            "emotion": emotion,
            "confidence": confidence,
            "response_style": self._determine_response_style(emotion, confidence, input_length)
        }
        
        return context
    
    def _determine_response_style(self, emotion: str, confidence: float, input_length: int) -> str:
        """Determine the appropriate response style based on context"""
        if confidence > 0.8:
            if emotion in ["joy", "love"]:
                return "enthusiastic"
            elif emotion in ["sadness", "fear"]:
                return "gentle_supportive"
            elif emotion == "anger":
                return "calm_understanding"
            else:
                return "neutral_supportive"
        else:
            return "neutral_supportive" 
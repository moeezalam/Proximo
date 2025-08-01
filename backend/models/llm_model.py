import torch
import requests
import json
import logging

logger = logging.getLogger(__name__)

class MistralConversation:
    def __init__(self):
        self.ollama_model = "mistral:latest"
        self.ollama_api_url = "http://localhost:11434/api"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.max_length = 2048
        self.temperature = 0.7
        
    def load_model(self):
        """Load Mistral model from local Ollama installation"""
        try:
            logger.info("Loading Mistral model from local Ollama installation")
            
            # Verify Ollama is running and model is available
            response = requests.get(f"{self.ollama_api_url}/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Ollama is not running or accessible")
                raise ConnectionError("Ollama server not accessible. Please start Ollama first.")
                
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check for mistral model (could be mistral:latest or just mistral)
            available_mistral = [name for name in model_names if 'mistral' in name.lower()]
            
            if not available_mistral:
                logger.warning(f"No Mistral model found in Ollama. Available models: {model_names}")
                raise FileNotFoundError(f"Please install a Mistral model in Ollama first: 'ollama pull mistral'")
            
            # Use the first available mistral model
            self.ollama_model = available_mistral[0]
            logger.info(f"✅ Using Ollama model: {self.ollama_model}")
            self.model = "ollama_local"  # Flag that we're using Ollama
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            logger.error("Make sure Ollama is running: 'ollama serve'")
            raise
        except Exception as e:
            logger.error(f"Failed to load Ollama model: {str(e)}")
            raise
    def create_advanced_emotion_prompt(self, user_input: str, emotion_data: dict) -> str:
        """
        Create sophisticated emotion-aware system prompt using all RoBERTa data
        
        Args:
            user_input: The user's message
            emotion_data: Full emotion analysis from RoBERTa
        """
        
        primary_emotion = emotion_data.get("emotion", "neutral")
        confidence = emotion_data.get("confidence", 0.5)
        all_scores = emotion_data.get("all_scores", {})
        secondary_emotions = emotion_data.get("secondary_emotions", {})
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "very high"
            certainty_modifier = "clearly"
        elif confidence >= 0.6:
            confidence_level = "high" 
            certainty_modifier = "likely"
        elif confidence >= 0.4:
            confidence_level = "moderate"
            certainty_modifier = "somewhat"
        else:
            confidence_level = "low"
            certainty_modifier = "possibly"
        
        # Create emotion-specific response instructions
        emotion_instructions = {
            "joy": {
                "tone": "enthusiastic and warm",
                "approach": "Share in their happiness and amplify their positive energy",
                "avoid": "being overly subdued or bringing up negative topics",
                "language": "Use upbeat language, exclamation points, and celebratory words"
            },
            "sadness": {
                "tone": "gentle, compassionate, and understanding",
                "approach": "Offer comfort, validate their feelings, and provide emotional support",
                "avoid": "being overly cheerful or dismissive of their pain",
                "language": "Use soft, empathetic language and offer hope without minimizing their feelings"
            },
            "anger": {
                "tone": "calm, understanding, but not dismissive",
                "approach": "Acknowledge their frustration, help them process feelings constructively",
                "avoid": "escalating the situation or being confrontational",
                "language": "Use validating language while gently guiding toward resolution"
            },
            "fear": {
                "tone": "reassuring, stable, and supportive",
                "approach": "Provide comfort, help build confidence, offer practical support",
                "avoid": "dismissing their concerns or adding to their anxiety",
                "language": "Use calming, confident language that instills security"
            },
            "love": {
                "tone": "warm, appreciative, and emotionally receptive",
                "approach": "Acknowledge and appreciate their loving feelings warmly",
                "avoid": "being cold or overly analytical about their emotions",
                "language": "Use affectionate, warm language that honors their feelings"
            },
            "surprise": {
                "tone": "engaged, curious, and energetic",
                "approach": "Share in their sense of wonder and help explore their discovery",
                "avoid": "being bland or uninterested in their revelation",
                "language": "Use expressive language that shows fascination and interest"
            }
        }
        
        primary_instruction = emotion_instructions.get(primary_emotion, {
            "tone": "balanced and understanding",
            "approach": "Respond naturally while being attentive to their emotional state",
            "avoid": "being robotic or emotionally disconnected",
            "language": "Use natural, conversational language"
        })
        
        # Build secondary emotion context
        secondary_context = ""
        if secondary_emotions:
            secondary_list = [f"{emotion} ({score:.0%})" for emotion, score in secondary_emotions.items()]
            secondary_context = f"\n- Secondary emotions detected: {', '.join(secondary_list)}"
            secondary_context += "\n- Consider these underlying feelings in your response"
        
        # Create the comprehensive system prompt for Ollama
        system_prompt = f"""You are Proximo, an emotionally intelligent AI companion with advanced empathy capabilities. You have been given detailed emotional analysis of the user's message and must respond with perfect emotional attunement.

EMOTIONAL ANALYSIS:
- Primary emotion: {primary_emotion} ({certainty_modifier} detected, {confidence:.0%} confidence)
- Confidence level: {confidence_level}{secondary_context}

RESPONSE GUIDELINES:
- Tone: {primary_instruction['tone']}
- Approach: {primary_instruction['approach']}
- Avoid: {primary_instruction['avoid']}
- Language style: {primary_instruction['language']}

ADVANCED INSTRUCTIONS:
1. Emotional Mirroring: Match approximately 70% of their emotional intensity
2. Validation: Always acknowledge their emotional state before responding to content
3. Progression: Guide the conversation toward emotional growth or resolution
4. Authenticity: Sound natural and human-like, not clinical or robotic
5. Brevity: Keep responses conversational (2-4 sentences typically)

CONFIDENCE ADJUSTMENT:
- High confidence ({confidence:.0%}): Respond decisively to the detected emotion
- Lower confidence: Acknowledge uncertainty and gently probe for clarification

Remember: You are a caring friend who truly understands emotions, not a therapist or counselor.

User's message: "{user_input}"

Respond as Proximo with perfect emotional intelligence:"""

        return system_prompt
    def generate_response(self, user_input: str, emotion_data: dict) -> str:
        """
        Generate emotion-aware response using comprehensive emotion data via Ollama
        
        Args:
            user_input: User's message
            emotion_data: Complete emotion analysis from RoBERTa
        """
        if not self.model:
            self.load_model()
            
        try:
            # Create advanced emotion-aware prompt
            prompt = self.create_advanced_emotion_prompt(user_input, emotion_data)
            
            # Get emotion-tuned parameters
            emotion = emotion_data.get("emotion", "neutral")
            confidence = emotion_data.get("confidence", 0.5)
            temp, top_p = self.get_emotion_parameters(emotion, confidence)
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_api_url}/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "top_p": top_p,
                        "num_predict": 120,  # Limit response length
                        "repeat_penalty": 1.1
                    }
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Clean up response
                response_text = self.clean_response(response_text, emotion)
                
                return response_text
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self.get_fallback_response(emotion)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return self.get_fallback_response(emotion_data.get("emotion", "neutral"))
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self.get_fallback_response(emotion_data.get("emotion", "neutral"))

    def get_emotion_parameters(self, emotion: str, confidence: float):
        """
        Get generation parameters tuned for specific emotions
        
        Returns: (temperature, top_p)
        """
        base_params = {
            "joy": (0.8, 0.9),      # More creative and energetic
            "sadness": (0.6, 0.8),  # More gentle and measured
            "anger": (0.7, 0.85),   # Balanced but assertive
            "fear": (0.65, 0.8),    # Careful and measured
            "love": (0.75, 0.9),    # Warm and expressive
            "surprise": (0.85, 0.95) # Very expressive and dynamic
        }
        
        temp, top_p = base_params.get(emotion, (0.7, 0.85))
        
        # Adjust based on confidence
        if confidence < 0.5:
            temp *= 0.9  # Be more conservative with uncertain emotions
            
        return temp, top_p

    def clean_response(self, response: str, emotion: str) -> str:
        """Clean and optimize the response"""
        # Remove any remaining prompt artifacts
        response = response.replace("Proximo:", "").strip()
        
        # Remove excessive repetition
        sentences = response.split('. ')
        unique_sentences = []
        for sentence in sentences:
            if sentence.strip() not in unique_sentences:
                unique_sentences.append(sentence.strip())
        
        response = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            if emotion in ['joy', 'surprise']:
                response += '!'
            else:
                response += '.'
        
        return response

    def get_fallback_response(self, emotion: str) -> str:
        """Provide fallback responses for errors"""
        fallbacks = {
            "joy": "I can sense your happiness! That's wonderful to hear.",
            "sadness": "I can feel that you're going through something difficult. I'm here for you.",
            "anger": "I understand you're frustrated right now. Those feelings are valid.",
            "fear": "I can sense some worry in your message. You're not alone in this.",
            "love": "I can feel the warmth in your words. That's really beautiful.",
            "surprise": "I can tell something unexpected happened! That must be quite something."
        }
        
        return fallbacks.get(emotion, "I hear you, and I'm here to listen. Could you tell me more?")
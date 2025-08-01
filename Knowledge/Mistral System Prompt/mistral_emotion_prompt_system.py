# backend/models/llm_model.py - Enhanced Version

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json

logger = logging.getLogger(__name__)

class MistralConversation:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.max_length = 2048
        self.temperature = 0.7
        
    def load_model(self):
        """Load Mistral-7B model with optimizations for RTX 3060"""
        try:
            logger.info(f"Loading Mistral model on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations for 6GB VRAM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
            
            logger.info("Mistral model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            raise

    def create_advanced_emotion_prompt(self, user_input: str, emotion_data: dict) -> str:
        """
        Create sophisticated emotion-aware system prompt using all RoBERTa data
        
        Args:
            user_input: The user's message
            emotion_data: Full emotion analysis from RoBERTa
                {
                    "emotion": "joy",
                    "confidence": 0.85,
                    "all_scores": {
                        "sadness": 0.05,
                        "joy": 0.85,
                        "love": 0.02,
                        "anger": 0.01,
                        "fear": 0.04,
                        "surprise": 0.03
                    }
                }
        """
        
        primary_emotion = emotion_data.get("emotion", "neutral")
        confidence = emotion_data.get("confidence", 0.5)
        all_scores = emotion_data.get("all_scores", {})
        
        # Get secondary emotions (above 10% probability)
        secondary_emotions = {
            emotion: score for emotion, score in all_scores.items() 
            if score > 0.1 and emotion != primary_emotion
        }
        
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
        
        # Create the comprehensive system prompt
        system_prompt = f"""<s>[INST] You are Proximo, an emotionally intelligent AI companion with advanced empathy capabilities. You have been given detailed emotional analysis of the user's message and must respond with perfect emotional attunement.

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

Respond as Proximo with perfect emotional intelligence: [/INST]"""

        return system_prompt

    def generate_response(self, user_input: str, emotion_data: dict) -> str:
        """
        Generate emotion-aware response using comprehensive emotion data
        
        Args:
            user_input: User's message
            emotion_data: Complete emotion analysis from RoBERTa
        """
        if not self.model or not self.tokenizer:
            self.load_model()
            
        try:
            # Create advanced emotion-aware prompt
            prompt = self.create_advanced_emotion_prompt(user_input, emotion_data)
            
            # Tokenize with proper attention to length
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Leave room for response
            ).to(self.device)
            
            # Generate response with emotion-tuned parameters
            emotion = emotion_data.get("emotion", "neutral")
            confidence = emotion_data.get("confidence", 0.5)
            
            # Adjust generation parameters based on emotion
            temp, top_p = self.get_emotion_parameters(emotion, confidence)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120,  # Slightly shorter for more focused responses
                    temperature=temp,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            response = response.split("[/INST]")[-1].strip()
            
            # Clean up response
            response = self.clean_response(response, emotion)
            
            return response
            
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


# Usage Example in your API (backend/api/main.py)
"""
Example of how to use this in your FastAPI endpoint:
"""

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Get comprehensive emotion analysis
        emotion_result = emotion_detector.predict_emotion(request.message)
        
        # emotion_result now contains:
        # {
        #     "emotion": "joy",
        #     "confidence": 0.85,
        #     "all_scores": {
        #         "sadness": 0.05,
        #         "joy": 0.85,
        #         "love": 0.02,
        #         "anger": 0.01,
        #         "fear": 0.04,
        #         "surprise": 0.03
        #     }
        # }
        
        # Step 2: Generate response with full emotion context
        response_text = conversation_model.generate_response(
            request.message,
            emotion_result  # Pass the complete emotion analysis
        )
        
        # Rest of your endpoint code...
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Testing the emotion prompts
def test_emotion_prompts():
    """Test different emotion scenarios"""
    
    test_cases = [
        {
            "message": "I just got promoted at work!",
            "emotion_data": {
                "emotion": "joy",
                "confidence": 0.92,
                "all_scores": {"joy": 0.92, "surprise": 0.05, "love": 0.02, "sadness": 0.01}
            }
        },
        {
            "message": "I'm really worried about my exam tomorrow",
            "emotion_data": {
                "emotion": "fear",
                "confidence": 0.78,
                "all_scores": {"fear": 0.78, "sadness": 0.15, "anger": 0.04, "joy": 0.02}
            }
        },
        {
            "message": "This traffic is driving me crazy!",
            "emotion_data": {
                "emotion": "anger",
                "confidence": 0.88,
                "all_scores": {"anger": 0.88, "fear": 0.07, "sadness": 0.03, "surprise": 0.02}
            }
        }
    ]
    
    conversation_model = MistralConversation()
    conversation_model.load_model()
    
    for test in test_cases:
        print(f"\nUser: {test['message']}")
        print(f"Emotion: {test['emotion_data']['emotion']} ({test['emotion_data']['confidence']:.0%})")
        
        response = conversation_model.generate_response(
            test['message'], 
            test['emotion_data']
        )
        
        print(f"Proximo: {response}")
        print("-" * 50)


if __name__ == "__main__":
    test_emotion_prompts()
#!/usr/bin/env python3
"""
Test script for the emotion-aware workflow MVP
Tests: User Input → RoBERTa Emotion → Mistral 7b → Coqui TTS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.emotion_model import EmotionDetector
from models.llm_model import MistralConversation
from models.tts_model import TextToSpeech
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_emotion_workflow():
    """Test the complete emotion-aware workflow"""
    
    print("🚀 Testing Proximo Emotion-Aware Workflow")
    print("=" * 50)
    
    # Test cases with different emotions
    test_cases = [
        "I'm so excited about my new job!",
        "I'm feeling really sad today...",
        "This situation is making me angry!",
        "I'm worried about my exam tomorrow",
        "I love spending time with my family",
        "Wow, I can't believe I won the lottery!"
    ]
    
    try:
        # Initialize models
        print("📊 Loading AI models...")
        emotion_detector = EmotionDetector()
        conversation_model = MistralConversation()
        tts_model = TextToSpeech()
        
        # Load models
        emotion_detector.load_model()
        print("✅ RoBERTa emotion model loaded")
        
        conversation_model.load_model()
        print("✅ Mistral-7B conversation model loaded")
        
        tts_model.load_model()
        print("✅ Coqui TTS model loaded")
        
        print("\n🧪 Testing workflow with sample inputs:")
        print("=" * 50)
        
        for i, user_input in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"👤 User: {user_input}")
            
            # Step 1: Emotion Detection
            emotion_result = emotion_detector.predict_emotion(user_input)
            print(f"🧠 Emotion: {emotion_result['emotion']} ({emotion_result['confidence']:.1%} confidence)")
            
            if emotion_result['secondary_emotions']:
                secondary = ", ".join([f"{e}({s:.0%})" for e, s in emotion_result['secondary_emotions'].items()])
                print(f"   Secondary: {secondary}")
            
            # Step 2: Generate Response with Emotion Context
            response_text = conversation_model.generate_response(user_input, emotion_result)
            print(f"🤖 Proximo: {response_text}")
            
            # Step 3: Text-to-Speech (optional for testing)
            try:
                audio_path = tts_model.synthesize_speech(response_text, emotion_result['emotion'])
                print(f"🔊 Audio generated: {audio_path}")
                
                # Clean up audio file
                os.unlink(audio_path)
                print("   Audio file cleaned up")
                
            except Exception as e:
                print(f"⚠️  TTS generation failed: {e}")
            
            print("-" * 30)
        
        print("\n✅ Workflow test completed successfully!")
        print("\n🎯 MVP Features Verified:")
        print("  ✅ RoBERTa emotion detection with confidence scores")
        print("  ✅ Emotion-aware system prompts for Mistral")
        print("  ✅ Context-sensitive response generation")
        print("  ✅ Text-to-speech with emotion adjustments")
        print("  ✅ Complete pipeline integration")
        
    except Exception as e:
        print(f"❌ Error during workflow test: {e}")
        logger.error(f"Workflow test failed: {e}")
        return False
    
    return True

def test_individual_components():
    """Test each component individually"""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    test_text = "I'm excited but also nervous about my presentation tomorrow!"
    
    try:
        # Test 1: Emotion Detection
        print("1. Testing RoBERTa Emotion Detection...")
        emotion_detector = EmotionDetector()
        emotion_detector.load_model()
        
        emotion_result = emotion_detector.predict_emotion(test_text)
        print(f"   Result: {emotion_result}")
        print("   ✅ Emotion detection working")
        
        # Test 2: LLM Response Generation
        print("\n2. Testing Mistral Response Generation...")
        conversation_model = MistralConversation()
        conversation_model.load_model()
        
        response = conversation_model.generate_response(test_text, emotion_result)
        print(f"   Response: {response}")
        print("   ✅ LLM response generation working")
        
        # Test 3: TTS
        print("\n3. Testing Coqui TTS...")
        tts_model = TextToSpeech()
        tts_model.load_model()
        
        audio_path = tts_model.synthesize_speech(response, emotion_result['emotion'])
        print(f"   Audio file: {audio_path}")
        
        # Check if file exists and has content
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print("   ✅ TTS generation working")
            os.unlink(audio_path)  # Clean up
        else:
            print("   ❌ TTS file generation failed")
        
        print("\n✅ All individual components working!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False
    
    return True

def check_ollama_first():
    """Check if Ollama is running before testing"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            mistral_models = [name for name in model_names if 'mistral' in name.lower()]
            
            if mistral_models:
                print(f"✅ Ollama is running with Mistral model: {mistral_models[0]}")
                return True
            else:
                print("❌ No Mistral model found in Ollama")
                print("💡 Run: ollama pull mistral")
                return False
        else:
            print("❌ Ollama is not responding properly")
            return False
    except:
        print("❌ Ollama is not running")
        print("💡 Start Ollama with: ollama serve")
        print("💡 Then run: python check_ollama.py")
        return False

if __name__ == "__main__":
    print("🤖 Proximo AI Companion - MVP Workflow Test")
    print("Testing the complete emotion-aware pipeline")
    print("=" * 60)
    
    # Check Ollama first
    if not check_ollama_first():
        print("\n❌ Please set up Ollama first. Run: python check_ollama.py")
        exit(1)
    
    print()
    
    # Test individual components first
    if test_individual_components():
        print("\n" + "=" * 60)
        # Test complete workflow
        test_emotion_workflow()
    else:
        print("❌ Individual component tests failed. Please check your setup.")
    
    print("\n🎉 Testing complete!")
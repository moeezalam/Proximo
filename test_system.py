#!/usr/bin/env python3
import requests
import json
import time
import sys
import os

# Add backend to path
sys.path.append('backend')

def test_emotion_detection():
    """Test emotion detection with sample texts"""
    test_cases = [
        ("I'm so happy today!", "joy"),
        ("I feel really sad about this", "sadness"),
        ("This makes me so angry!", "anger"),
        ("I'm scared about tomorrow", "fear"),
        ("I love spending time with you", "love"),
        ("Wow, I didn't expect that!", "surprise")
    ]
    
    print("🧠 Testing Emotion Detection...")
    
    for text, expected_emotion in test_cases:
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": text, "include_voice": False},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                detected = data['emotion']
                confidence = data['confidence']
                
                status = "✅" if detected == expected_emotion else "⚠️"
                print(f"{status} '{text}' -> {detected} ({confidence:.2f}) [Expected: {expected_emotion}]")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing '{text}': {e}")
    
    print()

def test_conversation_flow():
    """Test conversation with emotion context"""
    print("💬 Testing Conversation Flow...")
    
    conversations = [
        "Hi, how are you?",
        "I'm feeling really excited about my new job!",
        "But I'm also nervous about starting something new.",
        "Can you help me feel more confident?"
    ]
    
    for i, message in enumerate(conversations, 1):
        try:
            print(f"User {i}: {message}")
            
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": message, "include_voice": False},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Proximo {i}: {data['response']}")
                print(f"Emotion: {data['emotion']} ({data['confidence']:.2f})")
                print()
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
def test_voice_generation():
    """Test TTS functionality"""
    print("🔊 Testing Voice Generation...")
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": "Hello, this is a voice test!", "include_voice": True},
            timeout=45
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('audio_base64'):
                print("✅ Voice generation successful!")
                print(f"Audio data length: {len(data['audio_base64'])} characters")
            else:
                print("⚠️ No audio data in response")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Voice test error: {e}")
    
    print()

def test_health():
    """Test system health"""
    print("🏥 Testing System Health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print("Models loaded:")
            for model, loaded in data['models_loaded'].items():
                status = "✅" if loaded else "❌"
                print(f"  {status} {model}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()

if __name__ == "__main__":
    print("🧪 Proximo AI System Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server to start...")
    for i in range(30):
        try:
            requests.get("http://localhost:8000/health", timeout=5)
            break
        except:
            time.sleep(2)
            print(f"Waiting... ({i+1}/30)")
    else:
        print("❌ Server not responding. Please start the backend first.")
        sys.exit(1)
    
    # Run tests
    test_health()
    test_emotion_detection()
    test_conversation_flow()
    test_voice_generation()
    
    print("🎉 Testing complete!")
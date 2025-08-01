#!/usr/bin/env python3
"""
Simple test to verify Ollama connection and Mistral model
"""

import requests
import json

def test_ollama_connection():
    """Test basic Ollama connection"""
    try:
        print("🔍 Testing Ollama connection...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            print(f"📋 Available models: {', '.join(model_names)}")
            
            # Find Mistral model
            mistral_models = [name for name in model_names if 'mistral' in name.lower()]
            if mistral_models:
                print(f"✅ Found Mistral model: {mistral_models[0]}")
                return mistral_models[0]
            else:
                print("❌ No Mistral model found")
                return None
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("💡 Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_mistral_generation(model_name):
    """Test Mistral text generation"""
    try:
        print(f"\n🧪 Testing {model_name} generation...")
        
        test_prompt = "You are Proximo, an AI companion. A user says: 'I'm excited about my new job!' Respond with enthusiasm."
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            if generated_text:
                print("✅ Mistral generation successful!")
                print(f"📝 Response: {generated_text}")
                return True
            else:
                print("❌ Empty response from Mistral")
                return False
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def main():
    print("🤖 Simple Ollama & Mistral Test")
    print("=" * 35)
    
    # Test connection and find model
    model_name = test_ollama_connection()
    
    if not model_name:
        print("\n❌ Cannot proceed without Mistral model")
        print("\n💡 To fix:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Install Mistral: ollama pull mistral")
        return False
    
    # Test generation
    if test_mistral_generation(model_name):
        print("\n🎉 Everything works!")
        print("✅ Ollama connection: OK")
        print("✅ Mistral model: OK")
        print("✅ Text generation: OK")
        print("\n🚀 Ready to run the full MVP!")
        return True
    else:
        print("\n❌ Generation test failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
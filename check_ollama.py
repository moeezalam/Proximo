#!/usr/bin/env python3
"""
Check if Ollama is running and has Mistral model available
"""

import requests
import json
import subprocess
import sys

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama service returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama service is not running")
        print("💡 Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_mistral_model():
    """Check if Mistral model is available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Look for any mistral model
            mistral_models = [name for name in model_names if 'mistral' in name.lower()]
            
            if mistral_models:
                print(f"✅ Mistral model(s) found: {', '.join(mistral_models)}")
                return True, mistral_models[0]
            else:
                print("❌ No Mistral model found in Ollama")
                print("💡 Install Mistral with: ollama pull mistral")
                print(f"📋 Available models: {', '.join(model_names) if model_names else 'None'}")
                return False, None
        else:
            print(f"❌ Could not get model list: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False, None

def test_mistral_generation(model_name):
    """Test if Mistral can generate a response"""
    try:
        test_prompt = "Hello! How are you today?"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            if generated_text:
                print(f"✅ Mistral generation test successful")
                print(f"📝 Test response: {generated_text[:100]}...")
                return True
            else:
                print("❌ Mistral returned empty response")
                return False
        else:
            print(f"❌ Mistral generation failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Mistral generation: {e}")
        return False

def main():
    print("🔍 Ollama & Mistral Setup Check")
    print("=" * 40)
    
    # Check if Ollama service is running
    if not check_ollama_service():
        print("\n💡 To start Ollama:")
        print("   1. Open a new terminal")
        print("   2. Run: ollama serve")
        print("   3. Keep that terminal open")
        return False
    
    print()
    
    # Check if Mistral model is available
    has_mistral, model_name = check_mistral_model()
    if not has_mistral:
        print("\n💡 To install Mistral:")
        print("   1. Run: ollama pull mistral")
        print("   2. Wait for download to complete")
        return False
    
    print()
    
    # Test Mistral generation
    if test_mistral_generation(model_name):
        print("\n🎉 Everything is ready!")
        print("✅ Ollama service is running")
        print(f"✅ Mistral model ({model_name}) is available")
        print("✅ Mistral can generate responses")
        print("\n🚀 You can now run:")
        print("   python test_emotion_workflow.py")
        print("   start_proximo.bat")
        return True
    else:
        print("\n❌ Mistral generation test failed")
        print("💡 Try restarting Ollama or reinstalling the model")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
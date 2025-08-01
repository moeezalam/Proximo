#!/usr/bin/env python3
"""
Check if all requirements are installed for the Proximo MVP
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({memory:.1f}GB)")
            return True
        else:
            print("⚠️  GPU not available - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")
        return False

def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            mistral_models = [name for name in model_names if 'mistral' in name.lower()]
            
            if mistral_models:
                print(f"✅ Ollama is running with Mistral: {mistral_models[0]}")
                return True
            else:
                print("⚠️  Ollama is running but no Mistral model found")
                print("💡 Install with: ollama pull mistral")
                return False
        else:
            print("⚠️  Ollama is not responding properly")
            return False
    except:
        print("⚠️  Ollama is not running")
        print("💡 Start with: ollama serve")
        return False

def main():
    print("🔍 Proximo MVP Requirements Check")
    print("=" * 40)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📦 Checking Python packages:")
    
    # Core packages for the workflow (removed transformers since we're using Ollama)
    required_packages = [
        ("torch", "torch"),
        ("TTS", "TTS"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("numpy", "numpy"),
        ("requests", "requests")
    ]
    
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_good = False
    
    print("\n🖥️  Checking hardware:")
    check_gpu()  # GPU is optional but recommended
    
    print("\n🤖 Checking Ollama setup:")
    ollama_ok = check_ollama()
    
    print("\n" + "=" * 40)
    
    if all_good and ollama_ok:
        print("✅ All requirements satisfied!")
        print("\n🚀 You can now run the MVP:")
        print("   1. Run: python test_emotion_workflow.py")
        print("   2. Or run: start_proximo.bat")
    else:
        print("❌ Some requirements are missing!")
        print("\n📋 To install missing packages:")
        print("   pip install torch TTS fastapi uvicorn numpy requests")
        print("   cd frontend && npm install")
        
        if not ollama_ok:
            print("\n🤖 To set up Ollama:")
            print("   1. Install Ollama from https://ollama.ai")
            print("   2. Run: ollama serve")
            print("   3. Run: ollama pull mistral")
            print("   4. Run: python check_ollama.py")
    
    return all_good and ollama_ok

if __name__ == "__main__":
    main()
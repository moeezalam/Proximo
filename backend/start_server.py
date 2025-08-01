#!/usr/bin/env python3
import uvicorn
import logging
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("🚀 Starting Proximo AI Companion Server...")
    print("📊 Loading AI models (this may take a few minutes)...")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent model reloading
        log_level="info"
    )
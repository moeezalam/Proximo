import os
import logging
from typing import Dict, Any

# Logging configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console"]
}

# Model configurations
MODEL_CONFIG = {
    "emotion": {
        "model_name": "mananshah296/roberta-emotion",
        "max_length": 512,
        "emotion_labels": ["sadness", "joy", "love", "anger", "fear", "surprise"]
    },
    "llm": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "max_length": 2048,
        "temperature": 0.7,
        "max_new_tokens": 150,
        "repetition_penalty": 1.1
    },
    "tts": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "language": "en",
        "voice_cloning": True
    }
}

# Server configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "log_level": "info",
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"]
}

# GPU configuration
GPU_CONFIG = {
    "memory_fraction": 0.9,
    "enable_flash_attention": True,
    "device_map": "auto",
    "torch_dtype": "float16"
}

# Emotion response styles
EMOTION_STYLES = {
    "joy": {
        "tone": "enthusiastic",
        "approach": "celebratory",
        "language": "positive and energetic"
    },
    "sadness": {
        "tone": "gentle",
        "approach": "supportive",
        "language": "comforting and understanding"
    },
    "anger": {
        "tone": "calm",
        "approach": "de-escalating",
        "language": "acknowledging and constructive"
    },
    "fear": {
        "tone": "reassuring",
        "approach": "supportive",
        "language": "comforting and confident"
    },
    "love": {
        "tone": "warm",
        "approach": "appreciative",
        "language": "caring and affectionate"
    },
    "surprise": {
        "tone": "excited",
        "approach": "engaging",
        "language": "wonder and curiosity"
    }
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_response_time": 10.0,  # seconds
    "max_audio_size_mb": 5.0,
    "min_confidence": 0.3,
    "max_input_length": 1000
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration"""
    return {
        "logging": LOGGING_CONFIG,
        "models": MODEL_CONFIG,
        "server": SERVER_CONFIG,
        "gpu": GPU_CONFIG,
        "emotion_styles": EMOTION_STYLES,
        "performance": PERFORMANCE_THRESHOLDS
    }

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get configuration for a specific model type"""
    return MODEL_CONFIG.get(model_type, {})

def get_server_config() -> Dict[str, Any]:
    """Get server configuration"""
    return SERVER_CONFIG

def get_gpu_config() -> Dict[str, Any]:
    """Get GPU configuration"""
    return GPU_CONFIG 
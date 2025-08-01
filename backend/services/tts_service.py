import logging
import base64
import os
from models.tts_model import TextToSpeech

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.tts_model = TextToSpeech()
        
    def load_model(self):
        """Load the TTS model"""
        try:
            self.tts_model.load_model()
            logger.info("TTS service model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS service model: {e}")
            raise
    
    def synthesize_speech(self, text: str, emotion: str = None) -> str:
        """
        Synthesize speech from text with emotion-aware adjustments
        Returns: str - Base64 encoded audio data
        """
        try:
            # Generate audio file
            audio_path = self.tts_model.synthesize_speech(text, emotion)
            
            # Convert to base64 for web transmission
            with open(audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode()
            
            # Clean up temporary file
            os.unlink(audio_path)
            
            logger.info(f"Speech synthesized successfully, audio length: {len(audio_base64)} chars")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    def set_speaker_voice(self, speaker_wav_path: str):
        """Set a custom speaker voice for voice cloning"""
        try:
            self.tts_model.set_speaker_voice(speaker_wav_path)
            logger.info(f"Speaker voice set: {speaker_wav_path}")
        except Exception as e:
            logger.error(f"Error setting speaker voice: {e}")
            raise
    
    def get_audio_info(self, audio_base64: str) -> dict:
        """
        Get information about the generated audio
        Returns: dict with audio information
        """
        try:
            # Decode base64 to get audio data
            audio_data = base64.b64decode(audio_base64)
            
            return {
                "size_bytes": len(audio_data),
                "size_mb": len(audio_data) / (1024 * 1024),
                "format": "WAV",
                "encoding": "base64"
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {
                "size_bytes": 0,
                "size_mb": 0,
                "format": "unknown",
                "encoding": "base64"
            } 
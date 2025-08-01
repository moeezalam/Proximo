import torch
from TTS.api import TTS
import torch  # ← ADD THIS IMPORT
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use a simpler TTS model that doesn't require speaker configuration
        self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        self.tts = None
        self.default_speaker_wav = None
        
    def load_model(self):
        """Load Coqui TTS model"""
        try:
            logger.info(f"Loading TTS model on {self.device}")
            self.tts = TTS(self.model_name).to(self.device)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            raise
    
    def set_speaker_voice(self, speaker_wav_path: str):
        """Set the speaker voice for cloning"""
        if os.path.exists(speaker_wav_path):
            self.default_speaker_wav = speaker_wav_path
            logger.info(f"Speaker voice set: {speaker_wav_path}")
        else:
            logger.warning(f"Speaker wav file not found: {speaker_wav_path}")
    
    def synthesize_speech(self, text: str, emotion: str = None) -> str:
        """
        Convert text to speech
        Returns: path to generated audio file
        """
        if not self.tts:
            self.load_model()
            
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Adjust speaking style based on emotion (optional enhancement)
            adjusted_text = self.adjust_text_for_emotion(text, emotion)
            
            # Generate speech with simpler model
            self.tts.tts_to_file(
                text=adjusted_text,
                file_path=output_path
            )
            
            logger.info(f"Speech synthesized: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    def adjust_text_for_emotion(self, text: str, emotion: str) -> str:
        """Adjust text delivery based on emotion (optional enhancement)"""
        if not emotion:
            return text
            
        # Add subtle modifications for emotional delivery
        emotion_adjustments = {
            "joy": text,  # Keep natural for joy
            "sadness": text.replace(".", "..."),  # Slower pacing
            "anger": text.upper() if len(text) < 50 else text,  # Emphasis for short angry texts
            "fear": text,  # Keep natural
            "love": text,  # Keep natural 
            "surprise": text + "!"  # Add excitement
        }
        
        return emotion_adjustments.get(emotion, text)

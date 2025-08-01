from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import logging
from typing import Optional
import asyncio
import base64

# Import our models
from models.emotion_model import EmotionDetector
from models.llm_model import MistralConversation  
from models.tts_model import TextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Proximo AI Companion", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
emotion_detector = EmotionDetector()
conversation_model = MistralConversation()
tts_model = TextToSpeech()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    include_voice: bool = True

class ChatResponse(BaseModel):
    response: str
    emotion: str
    confidence: float
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    logger.info("Loading AI models...")
    try:
        emotion_detector.load_model()
        conversation_model.load_model()
        tts_model.load_model()
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Proximo AI Companion API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "emotion": emotion_detector.model is not None,
            "conversation": conversation_model.model == "ollama_local",
            "tts": tts_model.tts is not None
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        # Step 1: Detect emotion
        emotion_result = emotion_detector.predict_emotion(request.message)
        logger.info(f"Detected emotion: {emotion_result['emotion']} ({emotion_result['confidence']:.2f})")
        
        # Step 2: Generate response with emotion context
        response_text = conversation_model.generate_response(
            request.message,
            emotion_result  # Pass complete emotion data
        )
        logger.info(f"Generated response: {response_text[:50]}...")
        
        # Step 3: Generate speech (if requested)
        audio_base64 = None
        audio_path = None
        
        if request.include_voice:
            try:
                audio_path = tts_model.synthesize_speech(response_text, emotion_result['emotion'])
                
                # Convert to base64 for web transmission
                with open(audio_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode()
                
                # Clean up temp file
                os.unlink(audio_path)
                
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                # Continue without audio
        
        return ChatResponse(
            response=response_text,
            emotion=emotion_result['emotion'],
            confidence=emotion_result['confidence'],
            audio_path=audio_path,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            include_voice = message_data.get("include_voice", True)
            
            # Process the message
            emotion_result = emotion_detector.predict_emotion(user_message)
            response_text = conversation_model.generate_response(
                user_message,
                emotion_result  # Pass complete emotion data
            )
            
            # Generate audio if requested
            audio_base64 = None
            if include_voice:
                try:
                    audio_path = tts_model.synthesize_speech(response_text, emotion_result['emotion'])
                    with open(audio_path, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode()
                    os.unlink(audio_path)
                except Exception as e:
                    logger.error(f"WebSocket TTS error: {e}")
            
            # Send response
            response = {
                "response": response_text,
                "emotion": emotion_result['emotion'],
                "confidence": emotion_result['confidence'],
                "audio_base64": audio_base64
            }
            
            await manager.send_personal_message(json.dumps(response), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
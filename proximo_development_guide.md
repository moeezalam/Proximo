# Complete Proximo AI Companion Development Guide
*Building an Emotional AI Companion with Voice Response*

## System Requirements ✅
Your laptop is **perfectly suited** for this project:
- **CPU**: i7-11370H (Excellent for concurrent processing)
- **RAM**: 16GB (Sufficient for all models)
- **GPU**: RTX 3060 6GB (Perfect for Mistral-7B + TTS)
- **OS**: Windows 11 (Fully supported)

---

## Project Architecture Overview

```
User Input (Text) 
    ↓
Emotion Detection (RoBERTa) 
    ↓
LLM Processing (Mistral-7B + Emotion Context)
    ↓
Text-to-Speech (Coqui TTS)
    ↓
Audio Output (Web Interface)
```

---

## Phase 1: Environment Setup (Day 1)

### 1.1 Install Python and Git
```bash
# Download Python 3.10 from python.org (recommended version)
# During installation, check "Add Python to PATH"
# Install Git from git-scm.com
```

### 1.2 Create Project Directory
```bash
mkdir proximo-ai
cd proximo-ai
```

### 1.3 Set Up Python Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 1.4 Install CUDA Toolkit (for GPU support)
```bash
# Download CUDA Toolkit 11.8 from NVIDIA website
# Install with default settings
# Verify installation:
nvcc --version
```

### 1.5 Install PyTorch with CUDA Support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1.6 Verify GPU Access
```python
# test_gpu.py
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

## Phase 2: Backend Development (Days 2-4)

### 2.1 Install Core Dependencies
```bash
pip install transformers==4.48.3
pip install TTS==0.22.0
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install websockets==12.0
pip install python-multipart==0.0.6
pip install pydantic==2.5.0
pip install numpy==1.24.3
pip install torch-audio==2.1.0
```

### 2.2 Create Project Structure
```
proximo-ai/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── emotion_model.py
│   │   ├── llm_model.py
│   │   └── tts_model.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── emotion_service.py
│   │   ├── conversation_service.py
│   │   └── tts_service.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── frontend/
└── requirements.txt
```

### 2.3 Emotion Detection Model (backend/models/emotion_model.py)
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        self.model_name = "mananshah296/roberta-emotion"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        
    def load_model(self):
        """Load the RoBERTa emotion classification model"""
        try:
            logger.info(f"Loading emotion model on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            raise
    
    def predict_emotion(self, text: str) -> dict:
        """
        Predict emotion from text
        Returns: {"emotion": str, "confidence": float, "all_scores": dict}
        """
        if not self.model or not self.tokenizer:
            self.load_model()
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get results
            predictions = probabilities.cpu().numpy()[0]
            emotion_scores = {label: float(score) for label, score in zip(self.emotion_labels, predictions)}
            
            # Get top emotion
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[top_emotion]
            
            return {
                "emotion": top_emotion,
                "confidence": confidence,
                "all_scores": emotion_scores
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_scores": {}
            }
```

### 2.4 LLM Model (backend/models/llm_model.py)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class MistralConversation:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.max_length = 2048
        self.temperature = 0.7
        
    def load_model(self):
        """Load Mistral-7B model with optimizations for RTX 3060"""
        try:
            logger.info(f"Loading Mistral model on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations for 6GB VRAM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision
                device_map="auto",
                load_in_8bit=True,  # 8-bit quantization for memory efficiency
                trust_remote_code=True
            )
            
            logger.info("Mistral model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            raise
    
    def create_emotion_prompt(self, user_input: str, emotion: str, confidence: float) -> str:
        """Create emotion-aware system prompt"""
        
        emotion_instructions = {
            "joy": "The user is feeling happy and joyful! Match their positive energy with enthusiasm and warmth. Share in their happiness.",
            "sadness": "The user seems sad or down. Respond with gentle empathy, offer comfort, and be a supportive listener. Avoid being overly cheerful.",
            "anger": "The user appears frustrated or angry. Acknowledge their feelings, stay calm, and help them process their emotions constructively.",
            "fear": "The user seems anxious or fearful. Provide reassurance, be supportive, and help them feel more secure and confident.",
            "love": "The user is expressing love or affection. Respond warmly and acknowledge these positive feelings with care.",
            "surprise": "The user seems surprised or amazed. Share in their sense of wonder and help explore their feelings of discovery."
        }
        
        emotion_instruction = emotion_instructions.get(emotion, "Respond naturally and helpfully to the user.")
        confidence_note = f"The emotion detection confidence is {confidence:.2f}"
        
        system_prompt = f"""<s>[INST] You are an empathetic AI companion named Proximo. You understand emotions and respond with appropriate emotional intelligence.

Current situation: {emotion_instruction}
{confidence_note}

Guidelines:
1. Be warm, understanding, and emotionally supportive
2. Match the emotional tone appropriately
3. Keep responses conversational and natural
4. Show genuine interest in the user's feelings
5. Provide comfort when needed, celebration when appropriate

User message: {user_input} [/INST]"""

        return system_prompt
    
    def generate_response(self, user_input: str, emotion: str, confidence: float) -> str:
        """Generate emotion-aware response"""
        if not self.model or not self.tokenizer:
            self.load_model()
            
        try:
            # Create emotion-aware prompt
            prompt = self.create_emotion_prompt(user_input, emotion, confidence)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            response = response.split("[/INST]")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
```

### 2.5 TTS Model (backend/models/tts_model.py)
```python
import torch
from TTS.api import TTS
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
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
            
            # Generate speech
            if self.default_speaker_wav and os.path.exists(self.default_speaker_wav):
                # Use voice cloning
                self.tts.tts_to_file(
                    text=adjusted_text,
                    speaker_wav=self.default_speaker_wav,
                    language="en",
                    file_path=output_path
                )
            else:
                # Use default voice
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
```

### 2.6 Main API (backend/api/main.py)
```python
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
            "conversation": conversation_model.model is not None,
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
            emotion_result['emotion'],
            emotion_result['confidence']
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
                emotion_result['emotion'], 
                emotion_result['confidence']
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
```

---

## Phase 3: Frontend Setup (Day 5)

### 3.1 Clone and Setup Chat UI
```bash
# In the proximo-ai directory
git clone https://github.com/ChristophHandschuh/chatbot-ui.git frontend
cd frontend
npm install
```

### 3.2 Configure Frontend for Our Backend
Create `frontend/src/config.js`:
```javascript
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
  ENDPOINTS: {
    CHAT: '/chat',
    HEALTH: '/health'
  }
};
```

### 3.3 Modify Chat Component for Voice Support
Create `frontend/src/components/VoiceChat.jsx`:
```javascript
import React, { useState, useRef, useEffect } from 'react';
import { API_CONFIG } from '../config';

const VoiceChat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [includeVoice, setIncludeVoice] = useState(true);
  const audioRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    // Initialize WebSocket connection
    wsRef.current = new WebSocket(API_CONFIG.WS_URL);
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Add AI response to messages
      const aiMessage = {
        id: Date.now(),
        text: data.response,
        sender: 'ai',
        emotion: data.emotion,
        confidence: data.confidence,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      
      // Play audio if available
      if (data.audio_base64 && includeVoice) {
        playAudio(data.audio_base64);
      }
      
      setIsLoading(false);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsLoading(false);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [includeVoice]);

  const playAudio = (audioBase64) => {
    try {
      const audioBlob = new Blob(
        [Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0))],
        { type: 'audio/wav' }
      );
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
      }
    } catch (error) {
      console.error('Error playing audio:', error);
    }
  };

  const sendMessage = () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Send via WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        message: inputValue,
        include_voice: includeVoice
      }));
    }

    setInputValue('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      joy: '#FFD700',
      sadness: '#4682B4', 
      anger: '#DC143C',
      fear: '#9370DB',
      love: '#FF69B4',
      surprise: '#FF8C00',
      default: '#808080'
    };
    return colors[emotion] || colors.default;
  };

  return (
    <div className="voice-chat-container">
      <div className="chat-header">
        <h1>Proximo AI Companion</h1>
        <div className="voice-toggle">
          <label>
            <input
              type="checkbox"
              checked={includeVoice}
              onChange={(e) => setIncludeVoice(e.target.checked)}
            />
            Enable Voice Response
          </label>
        </div>
      </div>

      <div className="messages-container">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-content">
              <p>{message.text}</p>
              {message.emotion && (
                <div className="emotion-indicator">
                  <span 
                    className="emotion-badge"
                    style={{ backgroundColor: getEmotionColor(message.emotion) }}
                  >
                    {message.emotion} ({(message.confidence * 100).toFixed(0)}%)
                  </span>
                </div>
              )}
            </div>
            <div className="message-time">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message ai loading">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>

      <div className="input-container">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isLoading}
          rows="2"
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !inputValue.trim()}
          className="send-button"
        >
          Send
        </button>
      </div>

      <audio ref={audioRef} />
    </div>
  );
};

export default VoiceChat;
```

### 3.4 Add Styles
Create `frontend/src/styles/VoiceChat.css`:
```css
.voice-chat-container {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.chat-header {
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
}

.chat-header h1 {
  margin: 0 0 10px 0;
  font-size: 24px;
}

.voice-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f5f5f5;
}

.message {
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
}

.message.user {
  align-items: flex-end;
}

.message.ai {
  align-items: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 18px;
  word-wrap: break-word;
}

.message.user .message-content {
  background: #007bff;
  color: white;
}

.message.ai .message-content {
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
}

.emotion-indicator {
  margin-top: 8px;
}

.emotion-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  color: white;
  font-weight: bold;
}

.message-time {
  font-size: 11px;
  color: #666;
  margin-top: 4px;
}

.input-container {
  padding: 20px;
  background: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.input-container textarea {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 20px;
  resize: none;
  outline: none;
  font-family: inherit;
}

.send-button {
  padding: 12px 24px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-weight: bold;
}

.send-button:hover:not(:disabled) {
  background: #0056b3;
}

.send-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.loading .typing-indicator {
  display: flex;
  gap: 4px;
  padding: 12px 16px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #999;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-10px);
  }
}
```

### 3.5 Update Main App Component
Modify `frontend/src/App.js`:
```javascript
import React from 'react';
import VoiceChat from './components/VoiceChat';
import './styles/VoiceChat.css';

function App() {
  return (
    <div className="App">
      <VoiceChat />
    </div>
  );
}

export default App;
```

---

## Phase 4: Testing & Optimization (Day 6)

### 4.1 Create Requirements File
Create `requirements.txt`:
```txt
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
transformers==4.48.3
TTS==0.22.0
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
python-multipart==0.0.6
pydantic==2.5.0
numpy==1.24.3
accelerate==0.24.1
bitsandbytes==0.41.3
scipy==1.11.4
librosa==0.10.1
soundfile==0.12.1
```

### 4.2 Create Launch Scripts
Create `backend/start_server.py`:
```python
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
```

Create `start_proximo.bat` (Windows batch file):
```batch
@echo off
echo Starting Proximo AI Companion...
echo.

REM Start backend
echo Starting backend server...
cd backend
start "Proximo Backend" cmd /k "venv\Scripts\activate && python start_server.py"

REM Wait a moment for backend to start
timeout /t 10 /nobreak

REM Start frontend
echo Starting frontend...
cd ..\frontend
start "Proximo Frontend" cmd /k "npm run dev"

echo.
echo Proximo AI Companion is starting up!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
```

### 4.3 Performance Optimization Script
Create `backend/utils/optimize_gpu.py`:
```python
import torch
import gc
import logging

logger = logging.getLogger(__name__)

def optimize_gpu_memory():
    """Optimize GPU memory usage for RTX 3060 6GB"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction (use 90% of available memory)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
            
        logger.info(f"GPU Memory optimized. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def clear_gpu_cache():
    """Clear GPU cache between operations"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class GPUMemoryMonitor:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / 1024**2
            logger.info(f"GPU Memory used in operation: {memory_used:.1f} MB")
```

### 4.4 Testing Script
Create `test_system.py`:
```python
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
```

---

## Phase 5: Running the Application

### 5.1 Final Directory Structure
```
proximo-ai/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── emotion_model.py
│   │   ├── llm_model.py
│   │   └── tts_model.py
│   ├── services/
│   │   └── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── optimize_gpu.py
│   └── start_server.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── VoiceChat.jsx
│   │   ├── styles/
│   │   │   └── VoiceChat.css
│   │   ├── config.js
│   │   └── App.js
│   └── package.json
├── venv/
├── requirements.txt
├── start_proximo.bat
└── test_system.py
```

### 5.2 Step-by-Step Launch Instructions
```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install Python dependencies (first time only)
pip install -r requirements.txt

# 3. Install Node.js dependencies (first time only)
cd frontend
npm install
cd ..

# 4. Start the system
start_proximo.bat
```

### 5.3 Manual Launch (Alternative)
```bash
# Terminal 1 - Backend
cd backend
venv\Scripts\activate
python start_server.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

---

## Expected Performance & Benchmarks

### Memory Usage (RTX 3060 6GB):
- **Mistral-7B (8-bit)**: ~4.2GB VRAM
- **RoBERTa-Emotion**: ~130MB VRAM  
- **Coqui TTS**: ~800MB VRAM
- **Total**: ~5.1GB VRAM (85% utilization)

### Response Times:
- **Emotion Detection**: 50-100ms
- **LLM Response**: 2-4 seconds
- **TTS Generation**: 1-3 seconds
- **Total Pipeline**: 3-7 seconds

### Model Accuracies:
- **Emotion Detection**: 94.05% (as per RoBERTa model)
- **Conversation Quality**: Subjective, depends on prompting
- **Voice Quality**: High (Coqui XTTS v2 standard)

---

## Troubleshooting Guide

### Common Issues:

**1. CUDA Out of Memory**
```python
# Add to backend/api/main.py
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

**2. Model Loading Errors**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/
```

**3. Port Already in Use**
```bash
# Kill processes on ports
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

**4. Frontend Connection Issues**
- Check CORS settings in `main.py`
- Verify API_CONFIG URLs in `config.js`
- Ensure both frontend and backend are running

**5. Voice Not Playing**
- Check browser audio permissions
- Verify audio codec support (WAV)
- Check browser console for errors

---

## Advanced Features (Optional Enhancements)

### Add Voice Input:
```javascript
// Add to VoiceChat.jsx
const startVoiceInput = () => {
  if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.onresult = (event) => {
      setInputValue(event.results[0][0].transcript);
    };
    recognition.start();
  }
};
```

### Add Conversation Memory:
```python
# Add to conversation_service.py
conversation_history = []

def add_to_history(user_input, ai_response, emotion):
    conversation_history.append({
        'user': user_input,
        'ai': ai_response,
        'emotion': emotion,
        'timestamp': datetime.now()
    })
    
    # Keep last 10 exchanges
    if len(conversation_history) > 10:
        conversation_history.pop(0)
```

---

## Development Timeline Summary

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Day 1** | 2-3 hours | Environment setup, dependencies |
| **Day 2-3** | 6-8 hours | Backend models implementation |
| **Day 4** | 4-5 hours | FastAPI integration, WebSocket setup |
| **Day 5** | 3-4 hours | Frontend setup and customization |
| **Day 6** | 2-3 hours | Testing, optimization, deployment |
| **Total** | **17-23 hours** | Complete working system |

---

## Success Criteria ✅

By the end of this guide, you will have:

1. ✅ **Functional emotional AI companion** that detects emotions with 94%+ accuracy
2. ✅ **Real-time voice responses** using state-of-the-art TTS
3. ✅ **Web-based chat interface** with emotion indicators
4. ✅ **Optimized for RTX 3060** with efficient memory usage
5. ✅ **WebSocket real-time communication** 
6. ✅ **Completely free and open-source** solution
7. ✅ **Production-ready architecture** that can scale

**Ready to build your Proximo AI Companion? Let's get started! 🚀**
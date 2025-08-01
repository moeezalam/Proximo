# 🤖 Proximo AI Companion - MVP

**Emotion-Aware Chatbot with Voice Response**

## 🎯 Workflow

```
User Input → RoBERTa Emotion Detection → Mistral 7B (Emotion-Aware) → Coqui TTS → Voice Output
```

## ⚡ Quick Start

### 1. Set Up Ollama (Required)
```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama serve          # Keep this running in a terminal
ollama pull mistral   # Download Mistral model
```

### 2. Check Requirements
```bash
python check_requirements.py
python check_ollama.py
```

### 3. Test the Workflow
```bash
python test_emotion_workflow.py
```

### 4. Start the Full Application
```bash
# Windows
start_proximo.bat

# Or manually:
# Terminal 1: cd backend && python start_server.py
# Terminal 2: cd frontend && npm start
```

### 4. Access the App
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## 🧪 Testing the MVP

### Example Interactions:

**Input**: "I'm so excited about my new job!"
- **RoBERTa**: Detects `joy` (85% confidence)
- **Mistral**: Responds with enthusiasm and warmth
- **TTS**: Generates excited voice tone

**Input**: "I'm feeling really sad today..."
- **RoBERTa**: Detects `sadness` (78% confidence)  
- **Mistral**: Responds with gentle empathy
- **TTS**: Generates comforting voice tone

**Input**: "This traffic is driving me crazy!"
- **RoBERTa**: Detects `anger` (82% confidence)
- **Mistral**: Responds calmly and constructively
- **TTS**: Generates measured voice tone

## 🔧 Technical Details

### Models Used:
- **Emotion**: `mananshah296/roberta-emotion` (94% accuracy)
- **LLM**: `mistral:latest` (via Ollama local installation)
- **TTS**: `tts_models/multilingual/multi-dataset/xtts_v2`

### Key Features:
- ✅ Real-time emotion detection with confidence scores
- ✅ Emotion-aware system prompts for contextual responses
- ✅ Secondary emotion detection for nuanced understanding
- ✅ Dynamic response parameters based on emotion type
- ✅ Voice synthesis with emotional adjustments
- ✅ WebSocket-based real-time communication

### Memory Usage:
- **Mistral-7B**: Runs via Ollama (separate process)
- **RoBERTa**: ~130MB VRAM
- **Coqui TTS**: ~800MB VRAM
- **Total Python Process**: ~1GB VRAM
- **Ollama Process**: Manages Mistral separately

## 📁 Project Structure

```
proximo-ai/
├── backend/
│   ├── models/
│   │   ├── emotion_model.py    # RoBERTa emotion detection
│   │   ├── llm_model.py        # Mistral conversation with emotion context
│   │   └── tts_model.py        # Coqui TTS voice synthesis
│   └── api/
│       └── main.py             # FastAPI server with WebSocket
├── frontend/
│   └── src/
│       └── components/
│           └── VoiceChat.jsx   # React chat interface
├── test_emotion_workflow.py    # Test the complete workflow
├── check_requirements.py       # Verify installation
└── start_proximo.bat          # Easy startup script
```

## 🎨 Emotion System Prompts

The MVP uses sophisticated emotion-aware prompts that:

1. **Analyze primary emotion** with confidence levels
2. **Consider secondary emotions** for nuanced responses  
3. **Adjust tone and approach** based on detected feelings
4. **Provide specific guidance** for each emotion type
5. **Tune generation parameters** (temperature, top_p) per emotion

### Example System Prompt:
```
You are Proximo, an emotionally intelligent AI companion...

EMOTIONAL ANALYSIS:
- Primary emotion: joy (clearly detected, 85% confidence)
- Secondary emotions detected: surprise (12%)

RESPONSE GUIDELINES:
- Tone: enthusiastic and warm
- Approach: Share in their happiness and amplify their positive energy
- Language style: Use upbeat language, exclamation points, and celebratory words

User's message: "I'm so excited about my new job!"
```

## 🚀 Next Steps

This MVP demonstrates the core workflow. Future enhancements could include:

- **Voice Input**: Speech-to-text for complete voice interaction
- **Conversation Memory**: Context retention across messages
- **Personality Customization**: Different AI personas
- **Emotion Visualization**: Real-time emotion graphs
- **Multi-language Support**: International emotion detection

## 🛠️ Troubleshooting

### Common Issues:

**GPU Memory Error**:
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Model Loading Fails**:
```bash
# Check internet connection and try again
# Models are downloaded automatically on first run
```

**Port Already in Use**:
```bash
# Kill processes on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

---

**🎉 Enjoy your emotion-aware AI companion!**

The MVP successfully demonstrates how RoBERTa emotion detection can enhance conversational AI through contextual system prompts, creating more empathetic and engaging interactions.
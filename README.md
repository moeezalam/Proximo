# 🤖 Proximo AI Companion - Complete Setup Guide

**Emotion-Aware Chatbot with Voice Response**

An emotionally intelligent AI companion that provides personalized, empathetic responses with voice output. Built with open-source models and optimized for consumer hardware.

## 🎯 Project Overview

### Workflow
```
User Input → RoBERTa Emotion Detection → Mistral 7B (Emotion-Aware) → Coqui TTS → Voice Output
```

### Key Features
- ✅ **Real-time emotion detection** with 94%+ accuracy using RoBERTa
- ✅ **Emotion-aware responses** using Mistral 7B with sophisticated system prompts
- ✅ **Natural voice synthesis** using Coqui TTS
- ✅ **WebSocket-based real-time chat** interface
- ✅ **Secondary emotion detection** for nuanced understanding
- ✅ **Privacy-first**: 100% local processing, no data leaves your device

---

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Intel i7 or AMD Ryzen 7 (or equivalent)
- **RAM**: 16GB+ (recommended 32GB for optimal performance)
- **GPU**: NVIDIA RTX 3060 6GB+ (CUDA support) - Optional but recommended
- **Storage**: 10GB+ free space for models and dependencies
- **Internet**: Required for initial model downloads

### Software Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.10.x (REQUIRED - other versions may cause compatibility issues)
- **Node.js**: 16.x or 18.x
- **Git**: Latest version
- **Ollama**: Latest version (for local Mistral model)

---

## 🚀 Complete Installation Guide

### Step 1: Install System Dependencies

#### 1.1 Install Python 3.10
**Windows:**
1. Download Python 3.10.x from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   # Should show: Python 3.10.x
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

#### 1.2 Install Node.js
Download and install Node.js 18.x from [nodejs.org](https://nodejs.org/)

Verify installation:
```bash
node --version  # Should show v18.x.x
npm --version   # Should show 9.x.x or higher
```

#### 1.3 Install Git
Download and install Git from [git-scm.com](https://git-scm.com/)

#### 1.4 Install Ollama
1. Download Ollama from [ollama.ai](https://ollama.ai/)
2. Install following the platform-specific instructions
3. Verify installation:
   ```bash
   ollama --version
   ```

### Step 2: Clone and Setup Project

#### 2.1 Clone Repository
```bash
git clone <your-repository-url>
cd proximo-ai
```

#### 2.2 Create Python Virtual Environment
**IMPORTANT**: Use Python 3.10 specifically
```bash
# Windows
python -m venv venv310
venv310\Scripts\activate

# macOS/Linux
python3.10 -m venv venv310
source venv310/bin/activate
```

Verify you're using Python 3.10:
```bash
python --version  # Must show Python 3.10.x
```

### Step 3: Install Python Dependencies

#### 3.1 Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### 3.2 Install PyTorch with CUDA Support (Recommended)
**For NVIDIA GPU users:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only users:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 3.3 Install All Python Dependencies
```bash
pip install -r requirements.txt
```

**Complete requirements.txt contents:**
```
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
requests==2.31.0
```

### Step 4: Setup Ollama and Mistral Model

#### 4.1 Start Ollama Service
**Windows:**
```cmd
# Open a new terminal and keep it running
ollama serve
```

**macOS/Linux:**
```bash
# Open a new terminal and keep it running
ollama serve
```

#### 4.2 Install Mistral Model
**In a separate terminal:**
```bash
ollama pull mistral
```

This will download ~4GB of model data. Wait for completion.

#### 4.3 Verify Ollama Setup
```bash
python check_ollama.py
```

Expected output:
```
✅ Ollama service is running
✅ Mistral model(s) found: mistral:latest
✅ Mistral generation test successful
🎉 Everything is ready!
```

### Step 5: Install Frontend Dependencies

#### 5.1 Navigate to Frontend Directory
```bash
cd frontend
```

#### 5.2 Install Node.js Dependencies
```bash
npm install
```

**Complete package.json dependencies:**
```json
{
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.4",
    "@testing-library/react": "^13.3.0",
    "@testing-library/user-event": "^13.5.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^2.1.4"
  }
}
```

#### 5.3 Return to Project Root
```bash
cd ..
```

---

## 🧪 Testing Your Installation

### Test 1: Check All Requirements
```bash
python check_requirements.py
```

### Test 2: Test Ollama Connection
```bash
python test_ollama_simple.py
```

### Test 3: Test Complete Workflow
```bash
python test_emotion_workflow.py
```

Expected output should show:
- ✅ RoBERTa emotion detection working
- ✅ Mistral response generation working  
- ✅ Coqui TTS generation working
- ✅ Complete pipeline integration

---

## 🚀 Running the Application

### Method 1: Using Batch Script (Windows)
```cmd
.\start_proximo.bat
```

### Method 2: Manual Start

#### Terminal 1 - Backend Server
```bash
# Activate virtual environment
# Windows:
venv310\Scripts\activate
# macOS/Linux:
source venv310/bin/activate

# Start backend
cd backend
python start_server.py
```

#### Terminal 2 - Frontend Server
```bash
cd frontend
npm start
```

### Method 3: Individual Components
```bash
# Backend only
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend only
cd frontend
npm run dev
```

---

## 🌐 Accessing the Application

Once both servers are running:

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📁 Project Structure

```
proximo-ai/
├── 📁 backend/                    # Python FastAPI backend
│   ├── 📁 api/                   # API endpoints
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI app with WebSocket
│   ├── 📁 models/               # AI model implementations
│   │   ├── __init__.py
│   │   ├── emotion_model.py     # RoBERTa emotion detection
│   │   ├── llm_model.py         # Mistral conversation via Ollama
│   │   └── tts_model.py         # Coqui TTS voice synthesis
│   ├── 📁 services/             # Business logic layer
│   │   ├── __init__.py
│   │   ├── emotion_service.py
│   │   ├── conversation_service.py
│   │   └── tts_service.py
│   ├── 📁 utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── optimize_gpu.py
│   ├── package.json
│   └── start_server.py          # 🚀 MAIN BACKEND ENTRY POINT
├── 📁 frontend/                  # React frontend
│   ├── 📁 public/
│   │   └── index.html
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   │   └── VoiceChat.jsx    # Main chat interface
│   │   ├── 📁 styles/
│   │   │   └── VoiceChat.css    # Chat styling
│   │   ├── App.js               # Main React app
│   │   ├── index.js             # React entry point
│   │   ├── index.css            # Global styles
│   │   └── config.js            # API configuration
│   └── package.json             # Node.js dependencies
├── 📁 venv310/                   # Python virtual environment
├── 📁 Knowledge/                 # Documentation and guides
├── requirements.txt              # Python dependencies
├── check_requirements.py        # Requirements checker
├── check_ollama.py              # Ollama setup checker
├── test_ollama_simple.py        # Simple Ollama test
├── test_emotion_workflow.py     # Complete workflow test
├── start_proximo.bat            # Windows startup script
└── README.md                    # This file
```

---

## 🔧 Configuration Details

### Backend Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **WebSocket**: /ws endpoint
- **CORS**: Enabled for all origins (development)

### Frontend Configuration
- **Host**: localhost
- **Port**: 3000
- **API Base URL**: http://localhost:8000
- **WebSocket URL**: ws://localhost:8000/ws

### Model Configuration
- **Emotion Model**: `mananshah296/roberta-emotion`
- **LLM Model**: `mistral:latest` (via Ollama)
- **TTS Model**: `tts_models/en/ljspeech/tacotron2-DDC`

---

## 🎮 Using the Application

### 1. Basic Chat
1. Open http://localhost:3000
2. Type a message with emotion: "I'm so excited about my new job!"
3. Watch the emotion detection and response generation
4. Listen to the voice response (if enabled)

### 2. Testing Different Emotions

**Joy/Excitement:**
- "I'm so happy about this!"
- "This is amazing news!"
- "I can't wait for the weekend!"

**Sadness:**
- "I'm feeling really down today..."
- "This situation makes me sad"
- "I'm having a tough time"

**Anger:**
- "This is so frustrating!"
- "I'm really angry about this situation"
- "This makes me mad!"

**Fear/Anxiety:**
- "I'm worried about my exam tomorrow"
- "This makes me nervous"
- "I'm scared about the presentation"

**Love/Affection:**
- "I love spending time with my family"
- "My partner means everything to me"
- "I adore my pets"

**Surprise:**
- "I can't believe this happened!"
- "Wow, that's unexpected!"
- "I'm shocked by this news!"

### 3. Voice Controls
- Toggle voice response on/off using the checkbox
- Audio plays automatically when voice is enabled
- Responses are generated with emotional context

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Issues
**Problem**: "Module not found" or compatibility errors
**Solution**: 
```bash
python --version  # Must be 3.10.x
# If not, reinstall Python 3.10 and recreate virtual environment
```

#### 2. Ollama Not Running
**Problem**: "Ollama server not accessible"
**Solution**:
```bash
# Start Ollama in a separate terminal
ollama serve
# Then test
python check_ollama.py
```

#### 3. Mistral Model Missing
**Problem**: "No Mistral model found"
**Solution**:
```bash
ollama pull mistral
# Wait for download to complete
```

#### 4. GPU Memory Issues
**Problem**: CUDA out of memory
**Solution**:
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
# Or restart Python process to clear GPU cache
```

#### 5. Port Already in Use
**Problem**: "Port 8000 already in use"
**Solution**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

#### 6. TTS Model Download Issues
**Problem**: TTS model fails to download
**Solution**:
```bash
# Clear TTS cache and retry
rm -rf ~/.local/share/tts/
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

#### 7. Frontend Build Issues
**Problem**: npm install fails
**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### 8. Virtual Environment Issues
**Problem**: Wrong Python version in venv
**Solution**:
```bash
# Delete and recreate virtual environment
rm -rf venv310
python3.10 -m venv venv310
# Windows: venv310\Scripts\activate
# macOS/Linux: source venv310/bin/activate
pip install -r requirements.txt
```

### Performance Optimization

#### For GPU Users
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
# Should return True
```

#### For CPU Users
- Expect slower response times (10-30 seconds)
- Consider using a smaller TTS model
- Close other applications to free up RAM

### Memory Usage Guidelines

#### Minimum System (16GB RAM)
- Close unnecessary applications
- Use CPU-only mode if GPU has <6GB VRAM
- Expect 5-10 second response times

#### Recommended System (32GB RAM + RTX 3060+)
- Full GPU acceleration
- 2-5 second response times
- Can run multiple instances

---

## 🧪 Advanced Testing

### Test Individual Components

#### Test Emotion Detection Only
```python
from backend.models.emotion_model import EmotionDetector
detector = EmotionDetector()
detector.load_model()
result = detector.predict_emotion("I'm so excited!")
print(result)
```

#### Test Mistral via Ollama Only
```python
from backend.models.llm_model import MistralConversation
model = MistralConversation()
model.load_model()
response = model.generate_response("Hello!", {"emotion": "joy", "confidence": 0.8})
print(response)
```

#### Test TTS Only
```python
from backend.models.tts_model import TextToSpeech
tts = TextToSpeech()
tts.load_model()
audio_path = tts.synthesize_speech("Hello world!", "joy")
print(f"Audio saved to: {audio_path}")
```

### Load Testing
```bash
# Test multiple concurrent requests
python -c "
import requests
import concurrent.futures
import time

def test_request():
    response = requests.post('http://localhost:8000/chat', 
                           json={'message': 'Hello!', 'include_voice': False})
    return response.status_code

start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(test_request) for _ in range(10)]
    results = [future.result() for future in futures]

print(f'Completed 10 requests in {time.time() - start_time:.2f} seconds')
print(f'Success rate: {results.count(200)}/10')
"
```

---

## 📊 Performance Benchmarks

### Expected Response Times

#### With GPU (RTX 3060 6GB)
- **Emotion Detection**: 50-100ms
- **Mistral Response**: 1-3 seconds
- **TTS Generation**: 2-5 seconds
- **Total Pipeline**: 3-8 seconds

#### With CPU Only
- **Emotion Detection**: 200-500ms
- **Mistral Response**: 5-15 seconds
- **TTS Generation**: 10-30 seconds
- **Total Pipeline**: 15-45 seconds

### Memory Usage

#### Python Process
- **Base**: ~500MB
- **RoBERTa Loaded**: ~700MB
- **TTS Loaded**: ~1.5GB
- **Peak Usage**: ~2GB

#### Ollama Process
- **Mistral 7B**: ~4-6GB RAM
- **GPU VRAM**: ~4GB (if available)

#### Frontend Process
- **Node.js**: ~100-200MB
- **Browser**: ~200-500MB

---

## 🔒 Security Considerations

### Local Processing
- All AI processing happens locally
- No data sent to external servers
- Complete privacy protection

### Network Security
- Backend runs on localhost only by default
- CORS enabled for development (disable in production)
- No authentication required (add for production)

### Production Deployment
```python
# For production, modify backend/api/main.py:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🚀 Deployment Options

### Local Development (Current Setup)
- Perfect for testing and development
- Full feature access
- No external dependencies

### Docker Deployment
```dockerfile
# Dockerfile example (create if needed)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
EXPOSE 8000

CMD ["python", "backend/start_server.py"]
```

### Cloud Deployment Considerations
- Requires GPU instances for optimal performance
- Large model downloads (~10GB total)
- Consider model caching strategies
- Implement proper authentication

---

## 🤝 Contributing

### Development Setup
1. Follow installation guide above
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Run all tests: `python test_emotion_workflow.py`
5. Submit pull request

### Code Style
- Python: Follow PEP 8
- JavaScript: Use ESLint configuration
- Add type hints for Python functions
- Document all new features

### Testing Requirements
- All new features must include tests
- Maintain >90% test coverage
- Test on both GPU and CPU configurations

---

## 📄 License and Credits

### Open Source Models Used
- **RoBERTa-Emotion**: Apache 2.0 License
- **Mistral-7B**: Apache 2.0 License
- **Coqui TTS**: MPL 2.0 License

### Dependencies
- **FastAPI**: MIT License
- **React**: MIT License
- **PyTorch**: BSD License
- **Transformers**: Apache 2.0 License

### Attribution
This project uses open-source AI models and libraries. See individual package licenses for details.

---

## 📞 Support and Community

### Getting Help
1. Check this README first
2. Run diagnostic scripts: `python check_requirements.py`
3. Check troubleshooting section above
4. Create GitHub issue with:
   - System specifications
   - Error messages
   - Steps to reproduce

### Reporting Issues
Include the following information:
- Operating system and version
- Python version (`python --version`)
- GPU information (if applicable)
- Complete error traceback
- Steps to reproduce the issue

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Consider implementation complexity
- Check existing issues first

---

## 🎉 Congratulations!

If you've followed this guide completely, you now have a fully functional emotion-aware AI companion running locally on your machine! 

### What You've Built:
- ✅ **Advanced Emotion Detection**: 94%+ accuracy with RoBERTa
- ✅ **Intelligent Conversations**: Mistral 7B with emotion-aware prompts
- ✅ **Natural Voice Synthesis**: High-quality TTS with emotional context
- ✅ **Real-time Web Interface**: Modern React-based chat application
- ✅ **Complete Privacy**: 100% local processing

### Next Steps:
1. **Experiment**: Try different emotional inputs and observe responses
2. **Customize**: Modify system prompts for different personalities
3. **Extend**: Add new features like conversation memory
4. **Share**: Show off your emotion-aware AI companion!

**Enjoy your new AI companion! 🤖✨**

---

*Last updated: January 2025*
*Version: 1.0.0*
*Tested on: Windows 11, Python 3.10, Node.js 18*
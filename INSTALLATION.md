# Proximo AI Companion - Installation Guide

This guide will help you set up the Proximo AI Companion on your Windows system with RTX 3060.

## Prerequisites

### System Requirements
- **OS**: Windows 10/11
- **CPU**: Intel i7 or equivalent
- **RAM**: 16GB+ 
- **GPU**: NVIDIA RTX 3060 6GB+ (CUDA support required)
- **Storage**: 10GB+ free space for models

### Software Requirements
- **Python**: 3.10+ (3.13.5 recommended)
- **Node.js**: 16+ (for frontend)
- **Git**: Latest version
- **CUDA Toolkit**: 11.8 (for GPU support)

## Step-by-Step Installation

### 1. Install CUDA Toolkit

1. Download CUDA Toolkit 11.8 from [NVIDIA website](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Run the installer and follow the default settings
3. Verify installation:
   ```bash
   nvcc --version
   ```

### 2. Install Python Dependencies

```bash
# Navigate to project directory
cd Proximo

# Activate virtual environment
venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Install Node.js Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Return to project root
cd ..
```

### 4. Test GPU Access

```bash
# Test GPU availability
python test_gpu.py
```

Expected output:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3060 Laptop GPU
CUDA memory: 6.0 GB
```

### 5. Start the Application

#### Option A: Using the batch file (Recommended)
```bash
# From project root
start_proximo.bat
```

#### Option B: Manual startup
```bash
# Terminal 1 - Backend
cd backend
venv\Scripts\activate
python start_server.py

# Terminal 2 - Frontend
cd frontend
npm start
```

### 6. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

## Troubleshooting

### Common Issues

#### 1. PowerShell Execution Policy
**Error**: `npm : File cannot be loaded because running scripts is disabled`

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. CUDA Not Available
**Error**: `CUDA available: False`

**Solutions**:
- Install CUDA Toolkit 11.8
- Update NVIDIA drivers
- Check GPU compatibility

#### 3. Out of Memory Errors
**Error**: `CUDA out of memory`

**Solutions**:
- Close other GPU-intensive applications
- Reduce batch size in models
- Use CPU fallback (slower but functional)

#### 4. Model Download Issues
**Error**: `Error loading model`

**Solutions**:
- Check internet connection
- Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
- Use VPN if needed

#### 5. Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <PID_NUMBER> /F
```

### Performance Optimization

#### GPU Memory Management
The system automatically optimizes for RTX 3060:
- **Mistral-7B**: ~4.2GB VRAM (8-bit quantization)
- **RoBERTa-Emotion**: ~130MB VRAM
- **Coqui TTS**: ~800MB VRAM
- **Total**: ~5.1GB VRAM (85% utilization)

#### First Run
- Models will download on first run (may take 10-30 minutes)
- Subsequent runs will be faster
- Models are cached locally

## Testing

### Run System Tests
```bash
# Test all components
python test_system.py
```

### Manual Testing
1. Open http://localhost:3000
2. Type a message with emotion (e.g., "I'm so happy today!")
3. Check emotion detection accuracy
4. Verify voice generation works
5. Test WebSocket real-time communication

## Configuration

### Backend Configuration
Edit `backend/utils/config.py` to modify:
- Model parameters
- Server settings
- GPU optimization
- Performance thresholds

### Frontend Configuration
Edit `frontend/src/config.js` to change:
- API endpoints
- WebSocket URLs
- Feature toggles

## Model Information

### Emotion Detection
- **Model**: `mananshah296/roberta-emotion`
- **Accuracy**: 94.05%
- **Emotions**: Sadness, Joy, Love, Anger, Fear, Surprise
- **Size**: ~130MB

### Language Model
- **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimization**: 8-bit quantization
- **Size**: ~4.2GB VRAM
- **Features**: Emotion-aware prompts

### Text-to-Speech
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Features**: Voice cloning, multilingual
- **Size**: ~800MB VRAM
- **Output**: High-quality WAV

## Support

### Getting Help
1. Check this installation guide
2. Review the troubleshooting section
3. Run `python test_system.py` for diagnostics
4. Check logs in terminal output
5. Open an issue on GitHub

### Logs and Debugging
- Backend logs appear in the terminal
- Frontend logs appear in browser console (F12)
- Model loading progress is shown in terminal
- Error messages include detailed information

## Next Steps

After successful installation:
1. Test the system with various emotional inputs
2. Customize voice settings if desired
3. Explore the API documentation
4. Consider adding voice input features
5. Experiment with different conversation styles

---

**Happy chatting with Proximo! 🤖💬**
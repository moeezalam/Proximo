@echo off
echo 🤖 Starting Proximo AI Companion MVP
echo Emotion-Aware Chatbot with Voice Response
echo ==========================================
echo.

echo 📋 Workflow: User Input → RoBERTa Emotion → Mistral 7B (Ollama) → Coqui TTS
echo.

REM Check if Ollama is running
echo 🔍 Checking Ollama setup...
python check_ollama.py
if errorlevel 1 (
    echo.
    echo ❌ Ollama setup failed!
    echo Please ensure:
    echo   1. Ollama is installed and running: ollama serve
    echo   2. Mistral model is installed: ollama pull mistral
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "backend\venv" (
    echo ❌ Virtual environment not found!
    echo Please run: cd backend && python -m venv venv
    pause
    exit /b 1
)

REM Start backend
echo 🚀 Starting backend server...
cd backend
start "Proximo Backend" cmd /k "venv\Scripts\activate && python start_server.py"

REM Wait for backend to initialize
echo ⏳ Waiting for backend to load AI models (this may take 2-3 minutes)...
timeout /t 15 /nobreak

REM Start frontend
echo 🌐 Starting frontend...
cd ..\frontend
start "Proximo Frontend" cmd /k "npm start"

echo.
echo ✅ Proximo AI Companion is starting up!
echo 📊 Backend API: http://localhost:8000
echo 🌐 Frontend UI: http://localhost:3000
echo 🔍 Health Check: http://localhost:8000/health
echo.
echo 💡 Test the workflow:
echo    1. Type a message with emotion (e.g., "I'm excited!")
echo    2. Watch RoBERTa detect the emotion
echo    3. See Mistral respond with emotion-aware context
echo    4. Listen to Coqui TTS voice response
echo.
pause
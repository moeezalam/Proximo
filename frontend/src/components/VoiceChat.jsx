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
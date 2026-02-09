#!/bin/bash
# Start web_chat_app
echo "=== Web Chat App Start Script ==="

# Create necessary directories
mkdir -p uploads downloads conversations

# Check Python dependencies
echo "Checking dependencies..."
pip install -r requirements.txt -q

# Start application
echo "Starting Web server..."
echo "Please visit: http://localhost:5000"
python web_chat_app.py

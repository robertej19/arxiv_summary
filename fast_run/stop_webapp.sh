#!/bin/bash

# Stop 10-K Knowledge Base services

echo "🛑 Stopping 10-K Knowledge Base services..."

# Stop screen sessions
screen -S kb_api -X quit 2>/dev/null && echo "✅ API stopped" || echo "ℹ️  API was not running"
screen -S kb_frontend -X quit 2>/dev/null && echo "✅ Frontend stopped" || echo "ℹ️  Frontend was not running"

# Fallback: kill processes by pattern
pkill -f "python.*api.py" 2>/dev/null && echo "✅ API process killed"
pkill -f "streamlit" 2>/dev/null && echo "✅ Streamlit process killed"

echo "🏁 All services stopped"

#!/bin/bash

# Simple one-command startup for 10-K Knowledge Base
# Uses screen sessions to run both services in background

source .venv/bin/activate

echo "🚀 Starting 10-K Knowledge Base (Simple Mode)"
echo "============================================="

# Start API in screen session
screen -dmS kb_api bash -c 'source .venv/bin/activate && python api.py'
echo "✅ API started in screen session 'kb_api'"

# Wait for API to start
sleep 3

# Start Streamlit in screen session  
screen -dmS kb_frontend bash -c 'source .venv/bin/activate && python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true'
echo "✅ Frontend started in screen session 'kb_frontend'"

# Wait for services to initialize
sleep 5

echo ""
echo "🎉 Both services are running!"
echo "📊 Frontend: http://localhost:8501"
echo "🔧 API: http://localhost:8000"
echo ""
echo "To manage services:"
echo "  screen -r kb_api       # View API logs"
echo "  screen -r kb_frontend  # View frontend logs"  
echo "  screen -list           # List all sessions"
echo ""
echo "To stop services:"
echo "  screen -S kb_api -X quit"
echo "  screen -S kb_frontend -X quit"
echo "  # Or use: ./stop_webapp.sh"

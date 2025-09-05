#!/bin/bash

# One-liner startup using nohup (runs in background, logs to files)

source .venv/bin/activate && \
nohup python api.py > api.log 2>&1 & \
sleep 3 && \
nohup python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 & \
sleep 2 && \
echo "🚀 Services started!" && \
echo "📊 Frontend: http://localhost:8501" && \
echo "🔧 API: http://localhost:8000" && \
echo "📋 Logs: tail -f api.log streamlit.log" && \
echo "🛑 Stop: pkill -f 'python.*api.py' && pkill -f streamlit"

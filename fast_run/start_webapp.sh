#!/bin/bash

# 10-K Knowledge Base Web Application Startup Script
# Starts both the FastAPI backend and Streamlit frontend

set -e

echo "🚀 Starting 10-K Knowledge Base Web Application"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "api.py" ] || [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: Please run this script from the fast_run directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found at .venv"
    echo "   Please run setup first or create the virtual environment"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if 10-K database exists
if [ ! -f "10k_knowledge_base.db" ]; then
    echo "⚠️  Warning: 10-K knowledge base not found at 10k_knowledge_base.db"
    echo "   Some features may not work properly"
fi

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        echo "   Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "   Stopping Streamlit frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo "✅ Shutdown complete"
    exit 0
}

# Set up signal handlers for clean shutdown
trap cleanup SIGINT SIGTERM

echo ""
echo "🔧 Starting backend API server..."
python api.py &
API_PID=$!
echo "   API server started (PID: $API_PID)"

# Wait a moment for API to start
sleep 3

# Check if API is running
if ! curl -s http://localhost:8000/ >/dev/null 2>&1; then
    echo "❌ Error: API server failed to start"
    cleanup
fi

echo "✅ API server is running at http://localhost:8000"

echo ""
echo "🎨 Starting Streamlit frontend..."
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!
echo "   Streamlit frontend started (PID: $FRONTEND_PID)"

# Wait for Streamlit to start
echo "   Waiting for Streamlit to initialize..."
for i in {1..15}; do
    if curl -s http://localhost:8501/ >/dev/null 2>&1; then
        break
    fi
    echo "   ... checking ($i/15)"
    sleep 2
done

if curl -s http://localhost:8501/ >/dev/null 2>&1; then
    echo "✅ Streamlit frontend is running at http://localhost:8501"
else
    echo "⚠️  Streamlit may still be starting up at http://localhost:8501"
fi

echo ""
echo "🎉 Application is ready!"
echo "========================================"
echo "📊 Frontend (Streamlit): http://localhost:8501"
echo "🔧 Backend API:          http://localhost:8000"
echo "📚 API Documentation:    http://localhost:8000/docs"
echo ""
echo "Features available:"
echo "  • 🔍 Search 10-K filings"
echo "  • 🤖 AI-powered research"
echo "  • 📈 Analytics dashboard"
echo "  • 🏢 Company overviews"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"

# Keep the script running and wait for both processes
wait $API_PID $FRONTEND_PID
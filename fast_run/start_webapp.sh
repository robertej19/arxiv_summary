#!/bin/bash

echo "Starting 10-K Knowledge Base Web Application..."

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to get process info for a port
get_process_info() {
    local port=$1
    lsof -Pi :$port -sTCP:LISTEN | tail -n +2
}

# Install web dependencies
echo "Installing web dependencies..."
uv pip install -r requirements_web.txt

# Check if knowledge base exists
if [ ! -f "10k_knowledge_base.db" ]; then
    echo "Error: Knowledge base not found!"
    echo "Please run 'python build_10k_knowledge_base.py' first to create the database."
    exit 1
fi

# Check if FastAPI backend is already running (port 8000)
if check_port 8000; then
    echo "✅ FastAPI backend already running on port 8000"
    echo "   Process info: $(get_process_info 8000)"
    API_RUNNING=true
else
    echo "🔌 Starting FastAPI backend..."
    python api.py &
    API_PID=$!
    API_RUNNING=false
    sleep 3  # Wait for API to start
fi

# Check if Streamlit is already running (port 8501)
if check_port 8501; then
    echo "✅ Streamlit frontend already running on port 8501"
    echo "   Process info: $(get_process_info 8501)"
    STREAMLIT_RUNNING=true
else
    echo "📊 Starting Streamlit frontend..."
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    STREAMLIT_PID=$!
    STREAMLIT_RUNNING=false
    sleep 2  # Wait for Streamlit to start
fi

echo ""
echo "🚀 Web application status:"
echo ""

# Check final status
if check_port 8000; then
    echo "✅ Backend API (FastAPI): http://localhost:8000"
    echo "   📖 API Documentation: http://localhost:8000/docs"
else
    echo "❌ Backend API failed to start"
fi

if check_port 8501; then
    echo "✅ Frontend (Streamlit): http://localhost:8501"
else
    echo "❌ Frontend failed to start"
fi

echo ""

# Only set up cleanup for processes we started
if [ "$API_RUNNING" = false ] && [ "$STREAMLIT_RUNNING" = false ]; then
    echo "Press Ctrl+C to stop both services"
    
    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "Shutting down services..."
        [ ! -z "$API_PID" ] && kill $API_PID 2>/dev/null
        [ ! -z "$STREAMLIT_PID" ] && kill $STREAMLIT_PID 2>/dev/null
        echo "Services stopped."
        exit 0
    }
    
    # Set up signal handling
    trap cleanup SIGINT SIGTERM
    
    # Wait for processes
    wait $API_PID $STREAMLIT_PID

elif [ "$API_RUNNING" = false ]; then
    echo "Press Ctrl+C to stop the API service (Streamlit was already running)"
    
    cleanup() {
        echo ""
        echo "Shutting down API..."
        [ ! -z "$API_PID" ] && kill $API_PID 2>/dev/null
        echo "API stopped."
        exit 0
    }
    
    trap cleanup SIGINT SIGTERM
    wait $API_PID

elif [ "$STREAMLIT_RUNNING" = false ]; then
    echo "Press Ctrl+C to stop the Streamlit service (API was already running)"
    
    cleanup() {
        echo ""
        echo "Shutting down Streamlit..."
        [ ! -z "$STREAMLIT_PID" ] && kill $STREAMLIT_PID 2>/dev/null
        echo "Streamlit stopped."
        exit 0
    }
    
    trap cleanup SIGINT SIGTERM
    wait $STREAMLIT_PID

else
    echo "Both services were already running. Nothing to monitor."
    echo ""
    echo "To stop services manually:"
    echo "  - Kill processes using ports 8000 and 8501"
    echo "  - Or use: pkill -f 'api.py' && pkill -f 'streamlit'"
fi

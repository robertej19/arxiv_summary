#!/bin/bash

echo "10-K Knowledge Base Status Check"
echo "================================"

# Check if knowledge base exists
if [ -f "10k_knowledge_base.db" ]; then
    echo "✅ Knowledge base: Found"
    
    # Get database stats
    if command -v sqlite3 &> /dev/null; then
        filings_count=$(echo "SELECT COUNT(*) FROM filings;" | sqlite3 10k_knowledge_base.db 2>/dev/null || echo "N/A")
        echo "   📊 Filings in database: $filings_count"
    fi
else
    echo "❌ Knowledge base: Not found"
    echo "   Run: python build_10k_knowledge_base.py"
fi

echo ""

# Check API status (port 8000)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ FastAPI Backend: Running on port 8000"
    echo "   🔗 http://localhost:8000"
    echo "   📖 Docs: http://localhost:8000/docs"
    
    # Test API health
    if command -v curl &> /dev/null; then
        if curl -s http://localhost:8000/ >/dev/null 2>&1; then
            echo "   ✅ API responding"
        else
            echo "   ⚠️  API not responding"
        fi
    fi
else
    echo "❌ FastAPI Backend: Not running"
    echo "   Start with: python api.py"
fi

echo ""

# Check Streamlit status (port 8501)
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ Streamlit Frontend: Running on port 8501"
    echo "   🔗 http://localhost:8501"
    
    # Test Streamlit health
    if command -v curl &> /dev/null; then
        if curl -s http://localhost:8501/ >/dev/null 2>&1; then
            echo "   ✅ Frontend responding"
        else
            echo "   ⚠️  Frontend not responding"
        fi
    fi
else
    echo "❌ Streamlit Frontend: Not running"
    echo "   Start with: streamlit run streamlit_app.py"
fi

echo ""
echo "Quick Actions:"
echo "  🚀 Start everything: python launch.py"
echo "  🛑 Stop services: pkill -f 'api.py' && pkill -f 'streamlit'"
echo "  📊 Build database: python build_10k_knowledge_base.py"
```

```


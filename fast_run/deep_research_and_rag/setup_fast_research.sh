#!/bin/bash

# Setup script for fast 10-K research agents
# This will optimize your database and test the fast agents

echo "🚀 Setting up Fast 10-K Research Agents"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "tenk_research_agent.py" ]; then
    echo "❌ Error: Please run this script from the deep_research_and_rag directory"
    exit 1
fi

# Check if database exists
if [ ! -f "../10k_knowledge_base.db" ]; then
    echo "❌ Error: 10-K knowledge base not found at ../10k_knowledge_base.db"
    echo "Please build the knowledge base first:"
    echo "  cd .. && python build_10k_knowledge_base.py"
    exit 1
fi

echo "✅ Found 10-K knowledge base"

# Check if model exists
MODEL_PATHS=(
    "../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    "../../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)

MODEL_FOUND=false
for path in "${MODEL_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ Found model at $path"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    echo "⚠️  Warning: Model not found at expected locations"
    echo "To download the model, run: bash download_model.sh"
    echo "Continuing anyway (you can download it later)"
fi

# Optimize database for speed
echo ""
echo "🔧 Optimizing database for fast queries..."
python optimize_db_for_speed.py

if [ $? -eq 0 ]; then
    echo "✅ Database optimization complete"
else
    echo "❌ Database optimization failed"
    exit 1
fi

# Make scripts executable
chmod +x fast_tenk_agent.py
chmod +x ultra_fast_agent.py
chmod +x speed_test.py

echo ""
echo "🧪 Running quick speed test..."
python speed_test.py --quick

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Available fast research modes:"
echo "================================"
echo ""
echo "🚀 Ultra-Fast Mode (sub-second responses):"
echo "   python ultra_fast_agent.py 'What are AI risks?'"
echo "   - Uses pre-computed templates"
echo "   - Instant responses for common topics"
echo "   - Best for quick factual queries"
echo ""
echo "⚡ Fast Mode (1-3 second responses):"
echo "   python fast_tenk_agent.py 'How do companies approach AI?'"
echo "   - Dynamic database queries"
echo "   - Lightweight LLM synthesis"
echo "   - Good balance of speed and flexibility"
echo ""
echo "📊 Original Mode (30+ second responses):"
echo "   python tenk_research_agent.py 'Complex research question'"
echo "   - Deep multi-step research"
echo "   - Comprehensive evidence gathering"
echo "   - Best for detailed analysis"
echo ""
echo "🔍 Compare all modes:"
echo "   python speed_test.py"
echo ""
echo "💡 Tips:"
echo "   - Use ultra-fast for common topics (AI, cybersecurity, supply chain)"
echo "   - Use fast mode for most questions"
echo "   - Use original mode only when you need comprehensive analysis"
echo ""
echo "✅ Ready to use! Try the examples above."

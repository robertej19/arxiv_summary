#!/bin/bash

# Setup script for Lightning 10-K Research Agent
echo "🚀 Setting up Lightning 10-K Research Agent"
echo "=" * 50

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements_lightning.txt

# Download spaCy model
echo "🔤 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Check if 10-K database exists
if [ ! -f "../10k_knowledge_base.db" ]; then
    echo "⚠️  Warning: 10k_knowledge_base.db not found in parent directory"
    echo "   You'll need to run the data ingestion pipeline first"
    echo "   Or update the database path in the code"
fi

echo "✅ Setup complete!"
echo ""
echo "🏁 Quick start:"
echo "   1. Build knowledge base: python lightning_agent.py --build"
echo "   2. Test with question: python lightning_agent.py 'What is Apple's revenue?'"
echo "   3. Interactive mode: python lightning_agent.py --interactive"
echo "   4. Run benchmark: python lightning_agent.py --benchmark"

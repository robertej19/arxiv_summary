#!/bin/bash

echo "Setting up 10-K Knowledge Base..."

# Install dependencies using uv (as per memory)
echo "Installing dependencies..."
uv pip install -r requirements_kb.txt

# Build the knowledge base
echo "Building knowledge base from 10-K filings..."
python build_10k_knowledge_base.py

echo "Knowledge base setup complete!"
echo ""
echo "Usage examples:"
echo "  # Search for AI mentions"
echo "  python query_10k_kb.py --search 'artificial intelligence' --limit 5"
echo ""
echo "  # Get company overview"
echo "  python query_10k_kb.py --overview --ticker AAPL"
echo ""
echo "  # Compare risk factors across companies"
echo "  python query_10k_kb.py --compare risk_factors --tickers AAPL MSFT GOOGL"
echo ""
echo "  # Export chunks for RAG system"
echo "  python query_10k_kb.py --export-rag"

#!/bin/bash

# Download small, fast CPU model for 10-K research agent
MODEL_DIR="../models"
MODEL_NAME="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

echo "🤖 Setting up 10-K Research Agent Model"
echo "======================================="

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "✅ Model already exists at $MODEL_DIR/$MODEL_NAME"
    echo "   Size: $(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)"
else
    echo "📥 Downloading Qwen2.5-7B-Instruct (Q4_K_M quantized - ~4.4GB)"
    echo "   This is a fast, CPU-optimized model perfect for research tasks"
    echo ""
    
    # Download with progress
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
    else
        echo "❌ Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully!"
        echo "   Location: $MODEL_DIR/$MODEL_NAME"
        echo "   Size: $(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)"
    else
        echo "❌ Model download failed!"
        exit 1
    fi
fi

echo ""
echo "🚀 Ready to run 10-K Research Agent!"
echo "Usage: python tenk_research_agent.py 'Your research question here'"
```

```bash:/home/rober/arxiv_summarizer/fast_run/deep_research_and_rag/setup_agent.sh
#!/bin/bash

# Setup script for 10-K Research Agent
set -e

echo "🚀 Setting up 10-K Deep Research Agent"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "tenk_research_agent.py" ]; then
    echo "❌ Error: Please run this script from the deep_research_and_rag directory"
    exit 1
fi

# Check if 10-K database exists
DB_PATH="../10k_knowledge_base.db"
if [ ! -f "$DB_PATH" ]; then
    echo "❌ Error: 10-K knowledge base not found at $DB_PATH"
    echo "   Please run the knowledge base setup first:"
    echo "   cd .. && python build_10k_knowledge_base.py"
    exit 1
fi

echo "✅ 10-K database found at $DB_PATH"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv (fast installer)..."
    uv pip install -r requirements_agent.txt
else
    echo "Using pip..."
    pip install -r requirements_agent.txt
fi

# Download model if needed
echo ""
echo "🤖 Checking for language model..."
bash download_model.sh

# Test the setup
echo ""
echo "🧪 Testing setup..."
python -c "
import duckdb
from tools_10k import TenKSearchTool, TenKReadTool
from llama_cpp_model import LlamaCppModel
print('✅ All imports successful!')

# Test database connection
try:
    search_tool = TenKSearchTool()
    print('✅ 10-K database connection successful!')
except Exception as e:
    print(f'❌ Database error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "Usage examples:"
    echo "  python tenk_research_agent.py 'How are companies approaching AI?'"
    echo "  python tenk_research_agent.py 'What are the main cybersecurity risks?'"
    echo "  python tenk_research_agent.py 'How do tech companies view regulatory changes?'"
    echo ""
    echo "Environment variables:"
    echo "  LLAMA_GGUF=/path/to/model.gguf    # Custom model path"
    echo "  LLAMA_N_CTX=16384                 # Larger context window"
    echo "  LLAMA_MAX_TOK=1200                # More generation tokens"
else
    echo "❌ Setup failed during testing"
    exit 1
fi
```

Now let me create a quick demo/example script:

```python:/home/rober/arxiv_summarizer/fast_run/deep_research_and_rag/demo_10k_agent.py
#!/usr/bin/env python3
"""
Demo script for the 10-K Research Agent

Shows various research capabilities and example queries.
"""

import sys
from pathlib import Path
from tenk_research_agent import conduct_10k_research

# Example research questions for different use cases
DEMO_QUESTIONS = [
    {
        "title": "AI & Technology Strategy",
        "question": "How are major technology companies approaching artificial intelligence and what competitive advantages do they see?",
        "description": "Analyzes AI strategies across tech companies"
    },
    {
        "title": "Cybersecurity & Risk Management", 
        "question": "What are the primary cybersecurity risks that companies identify in their 10-K filings?",
        "description": "Reviews cybersecurity risk disclosures"
    },
    {
        "title": "Supply Chain & Manufacturing",
        "question": "How do companies describe supply chain disruptions and mitigation strategies?",
        "description": "Examines supply chain resilience approaches"
    },
    {
        "title": "ESG & Sustainability",
        "question": "What environmental and sustainability initiatives are companies reporting in their business strategies?",
        "description": "Reviews environmental strategy disclosures"
    },
    {
        "title": "Regulatory & Compliance",
        "question": "How do financial services companies describe regulatory changes and compliance costs?",
        "description": "Analyzes regulatory impact across financial sector"
    }
]

def run_demo_question(demo_item: dict):
    """Run a single demo research question."""
    print("\n" + "="*80)
    print(f"🔬 DEMO: {demo_item['title']}")
    print("="*80)
    print(f"Description: {demo_item['description']}")
    print(f"Question: {demo_item['question']}")
    print("="*80)
    
    try:
        answer = conduct_10k_research(demo_item['question'])
        print("\n" + "="*80)
        print("📋 RESEARCH FINDINGS")
        print("="*80)
        print(answer)
        print("="*80)
        return True
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        return False

def main():
    """Run demo of 10-K research agent."""
    
    print("🎭 10-K RESEARCH AGENT DEMO")
    print("="*50)
    print("This demo showcases the capabilities of the 10-K research agent")
    print("by running several example research questions.\n")
    
    if len(sys.argv) > 1:
        # Run specific demo by number
        try:
            demo_num = int(sys.argv[1])
            if 1 <= demo_num <= len(DEMO_QUESTIONS):
                demo_item = DEMO_QUESTIONS[demo_num - 1]
                success = run_demo_question(demo_item)
                return 0 if success else 1
            else:
                print(f"❌ Invalid demo number. Choose 1-{len(DEMO_QUESTIONS)}")
                return 1
        except ValueError:
            # Treat as custom question
            custom_question = " ".join(sys.argv[1:])
            print(f"🔬 CUSTOM RESEARCH QUESTION")
            print("="*50)
            answer = conduct_10k_research(custom_question)
            print("\n" + "="*50)
            print("📋 RESEARCH FINDINGS")
            print("="*50)
            print(answer)
            return 0
    
    # Interactive demo selection
    print("Available demos:")
    for i, demo in enumerate(DEMO_QUESTIONS, 1):
        print(f"  {i}. {demo['title']}")
        print(f"     {demo['description']}")
    
    print(f"\nChoose a demo (1-{len(DEMO_QUESTIONS)}) or 'q' to quit:")
    
    while True:
        try:
            choice = input("> ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("Goodbye! 👋")
                return 0
            
            demo_num = int(choice)
            if 1 <= demo_num <= len(DEMO_QUESTIONS):
                demo_item = DEMO_QUESTIONS[demo_num - 1]
                success = run_demo_question(demo_item)
                if success:
                    print("\n🎉 Demo completed!")
                
                print(f"\nChoose another demo (1-{len(DEMO_QUESTIONS)}) or 'q' to quit:")
            else:
                print(f"❌ Please choose a number between 1 and {len(DEMO_QUESTIONS)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            return 0

if __name__ == "__main__":
    sys.exit(main())
```

Finally, let me create a simple README for the research agent:

```markdown:/home/rober/arxiv_summarizer/fast_run/deep_research_and_rag/README_10K_AGENT.md
# 10-K Deep Research Agent

A specialized AI research agent that can quickly search through SEC 10-K filings and synthesize comprehensive answers about companies, industries, and business trends.

## Features

- **Fast Local LLM**: Uses Qwen2.5-7B quantized model optimized for CPU inference
- **Comprehensive Search**: Searches across all 10-K sections with intelligent query planning
- **Evidence Synthesis**: Combines findings from multiple companies and sources
- **Proper Citations**: Includes specific references to filing sections
- **Configurable**: Adjustable context window and generation parameters

## Quick Start

1. **Setup the agent**:
   ```bash
   cd deep_research_and_rag
   bash setup_agent.sh
   ```

2. **Run a research query**:
   ```bash
   python tenk_research_agent.py "How are companies approaching artificial intelligence?"
   ```

3. **Try the interactive demo**:
   ```bash
   python demo_10k_agent.py
   ```

## Usage Examples

### Basic Research
```bash
# Technology trends
python tenk_research_agent.py "What AI strategies do tech companies describe?"

# Risk analysis
python tenk_research_agent.py "What are the main cybersecurity risks companies face?"

# Financial analysis
python tenk_research_agent.py "How do companies describe inflation impacts?"
```

### Advanced Research
```bash
# Filter by specific aspects
python tenk_research_agent.py "How do financial services companies view regulatory changes?"

# Industry comparisons
python tenk_research_agent.py "Compare supply chain strategies across manufacturing companies"
```

## Configuration

Set environment variables to customize behavior:

```bash
# Use a different model
export LLAMA_GGUF=/path/to/your/model.gguf

# Increase context window (if your model supports it)
export LLAMA_N_CTX=16384

# Generate longer responses
export LLAMA_MAX_TOK=1200
```

## Architecture

The agent follows a structured research pipeline:

1. **Query Planning**: LLM generates 2-3 focused search queries
2. **Evidence Collection**: Searches 10-K database for relevant content
3. **Detailed Reading**: Reads full sections for the most promising results
4. **Synthesis**: LLM combines all evidence into a comprehensive answer

## Files

- `tenk_research_agent.py` - Main research agent
- `tools_10k.py` - DuckDB search and read tools  
- `demo_10k_agent.py` - Interactive demo with example questions
- `requirements_agent.txt` - Python dependencies
- `setup_agent.sh` - Automated setup script
- `download_model.sh` - Model download script

## Performance

- **Model Size**: ~4.4GB (Qwen2.5-7B Q4_K_M quantized)
- **Speed**: ~10-50 tokens/second on modern CPUs
- **Memory**: ~6-8GB RAM recommended
- **Research Time**: 30-60 seconds per question

## Troubleshooting

### Model Not Found
```bash
# Download the model manually
bash download_model.sh

# Or set custom path
export LLAMA_GGUF=/path/to/your/model.gguf
```

### Database Not Found
```bash
# Build the 10-K knowledge base first
cd ..
python build_10k_knowledge_base.py
```

### Memory Issues
```bash
# Reduce context window
export LLAMA_N_CTX=4096

# Use smaller max tokens
export LLAMA_MAX_TOK=512
```

## Integration

The agent can be easily integrated into other applications:

```python
from tenk_research_agent import conduct_10k_research

# Simple research
answer = conduct_10k_research("How do companies view climate risks?")

# Custom tools
from tools_10k import TenKSearchTool, TenKReadTool
search_tool = TenKSearchTool()
results = search_tool.forward("artificial intelligence", limit=10)
```
```

Now let's make the scripts executable and test the setup:

```


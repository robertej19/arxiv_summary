#!/usr/bin/env python3
"""
Demo script to show the enhanced citation linking functionality.

This script demonstrates how the new citation system works:
1. Citations are generated with unique IDs during research
2. Users can click on citations to view the original HTML source
3. The source is highlighted to show the exact quoted text
"""

import sys
import os

# Add the deep_research_and_rag directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))

from tenk_research_agent import conduct_10k_research

def demo_citation_linking():
    """Run a demo research query to show citation linking."""
    
    print("🔬 Citation Linking Demo")
    print("=" * 50)
    
    # Example research question
    question = "How are technology companies addressing artificial intelligence risks?"
    
    print(f"Research Question: {question}")
    print("\n" + "=" * 50)
    print("Running research with enhanced citation tracking...")
    print("=" * 50)
    
    # Run the research
    try:
        answer = conduct_10k_research(question)
        
        print("\n📋 RESEARCH FINDINGS WITH ENHANCED CITATIONS")
        print("=" * 50)
        print(answer)
        print("=" * 50)
        
        print("\n✨ CITATION ENHANCEMENT FEATURES:")
        print("- Each citation [1], [2], etc. now has a unique ID for tracking")
        print("- Citations link back to the exact location in the original HTML 10-K filing")
        print("- The original source text is highlighted to show the quoted content")
        print("- Users can click citations in the web interface to view sources")
        
        print("\n🚀 TO USE THE WEB INTERFACE:")
        print("1. Start the API server: python api.py")
        print("2. Start the Streamlit app: streamlit run streamlit_app.py")
        print("3. Navigate to 'AI Research' page")
        print("4. Ask a research question")
        print("5. Click on any citation [1], [2], etc. to view the source!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_citation_linking()

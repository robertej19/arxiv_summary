#!/usr/bin/env python3
"""
Example of how to use the streaming research API.

This demonstrates both the API endpoint and direct function usage.
"""

import requests
import json
import time

def example_api_streaming():
    """Example using the streaming API endpoint."""
    print("🌐 API Streaming Example")
    print("=" * 40)
    
    api_url = "http://localhost:8000/research/stream"
    question = "How do technology companies address cybersecurity risks?"
    
    try:
        response = requests.post(
            api_url,
            json={"question": question},
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        print(f"Question: {question}")
        print("-" * 40)
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    progress_percent = int(data.get("progress", 0) * 100)
                    print(f"[{progress_percent:3d}%] {data.get('step', 'unknown')}: {data.get('message', '')}")
                    
                    if data.get("step") == "completed":
                        print("\n✅ Research completed!")
                        final_answer = data.get("details", {}).get("final_answer", "")
                        print(f"Answer: {final_answer[:200]}...")
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        print("💡 Make sure the API server is running: python api.py")

def example_direct_streaming():
    """Example using the function directly."""
    print("\n🔧 Direct Function Example")
    print("=" * 40)
    
    import sys
    import os
    
    # Add the deep_research_and_rag module to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))
    
    try:
        from tenk_research_agent import conduct_10k_research_with_progress
        
        question = "What supply chain challenges do companies face?"
        print(f"Question: {question}")
        print("-" * 40)
        
        for update in conduct_10k_research_with_progress(question):
            progress_percent = int(update.progress * 100)
            print(f"[{progress_percent:3d}%] {update.step}: {update.message}")
            
            # Show some details for interesting steps
            if update.details and update.step in ["search_complete", "reading_complete"]:
                details = update.details
                if "evidence_count" in details:
                    print(f"         📊 Found {details['evidence_count']} evidence items")
                if "companies_found" in details:
                    print(f"         🏢 From {details['companies_found']} companies")
            
            if update.step == "completed":
                print("\n✅ Research completed!")
                break
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")

if __name__ == "__main__":
    print("🚀 Streaming Research Examples")
    print("=" * 50)
    
    # Try API streaming first
    example_api_streaming()
    
    # Then direct function
    example_direct_streaming()
    
    print("\n" + "=" * 50)
    print("📚 Usage Tips:")
    print("1. Start the API server: python api.py")
    print("2. Open the Streamlit app: streamlit run streamlit_app.py")
    print("3. Use the 'AI Research' page to see streaming in action")
    print("4. The progress bar and details will update in real-time!")

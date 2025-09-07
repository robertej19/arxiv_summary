#!/usr/bin/env python3
"""
Simple test for streaming functionality without full research pipeline.
"""

import sys
import os
import time

# Add the deep_research_and_rag module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))

def create_simple_progress_generator(question: str):
    """Create a simple progress generator for testing."""
    from tenk_research_agent import ProgressUpdate
    
    steps = [
        ("connected", "Connection established", 0.0),
        ("testing", "Testing streaming...", 0.2),
        ("progress1", "First progress update", 0.4),
        ("progress2", "Second progress update", 0.6),
        ("progress3", "Third progress update", 0.8),
        ("completed", "Test completed!", 1.0)
    ]
    
    for step, message, progress in steps:
        yield ProgressUpdate(
            step=step,
            message=message,
            progress=progress,
            details={"test": True, "question": question}
        )
        time.sleep(0.5)  # Simulate processing time

def test_simple_streaming():
    """Test basic streaming functionality."""
    print("🧪 Testing Simple Streaming...")
    print("=" * 40)
    
    test_question = "Test question"
    
    try:
        for update in create_simple_progress_generator(test_question):
            progress_percent = int(update.progress * 100)
            print(f"[{progress_percent:3d}%] {update.step}: {update.message}")
            
            # Test model_dump method
            data = update.model_dump()
            print(f"         JSON: {data}")
            
            if update.step == "completed":
                print("\n✅ Simple streaming test passed!")
                return True
                
    except Exception as e:
        print(f"❌ Simple streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_streaming():
    """Test API streaming endpoint."""
    print("\n🌐 Testing API Streaming...")
    print("=" * 40)
    
    import requests
    import json
    
    try:
        # Test basic API connection first
        response = requests.get("http://localhost:8000/sections", timeout=5)
        if response.status_code != 200:
            print("❌ API server not running or not accessible")
            return False
        
        print("✅ API server is accessible")
        
        # Test streaming endpoint
        response = requests.post(
            "http://localhost:8000/research/stream",
            json={"question": "simple test"},
            stream=True,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Streaming endpoint error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        print("✅ Streaming endpoint responding")
        
        # Read first few lines
        line_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    print(f"📦 Received: {data.get('step', 'unknown')} - {data.get('message', '')}")
                    line_count += 1
                    
                    if line_count >= 3 or data.get('step') == 'completed':
                        print("✅ API streaming test passed!")
                        return True
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error: {e}")
                    continue
            
            if line_count > 10:  # Safety limit
                break
                
        print("❌ No valid streaming data received")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Please start it with: python api.py")
        return False
    except Exception as e:
        print(f"❌ API streaming test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_simple_streaming()
    success2 = test_api_streaming()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Simple Streaming: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"API Streaming: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Streaming should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    sys.exit(0 if (success1 and success2) else 1)

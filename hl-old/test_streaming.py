#!/usr/bin/env python3
"""
Test script for streaming progress functionality.
"""

import sys
import os
import time

# Add the deep_research_and_rag module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))

def test_streaming_progress():
    """Test the streaming progress functionality."""
    try:
        from tenk_research_agent import conduct_10k_research_with_progress
        
        print("🧪 Testing streaming progress functionality...")
        print("=" * 50)
        
        test_question = "What are the main risks technology companies face?"
        
        print(f"Question: {test_question}")
        print("=" * 50)
        
        start_time = time.time()
        final_answer = None
        
        for update in conduct_10k_research_with_progress(test_question):
            elapsed = time.time() - start_time
            progress_percent = int(update.progress * 100)
            
            print(f"[{elapsed:6.1f}s] [{progress_percent:3d}%] {update.step}: {update.message}")
            
            if update.details:
                details = update.details
                if "queries" in details:
                    print(f"         📋 Queries: {len(details['queries'])}")
                if "evidence_count" in details:
                    print(f"         📊 Evidence: {details['evidence_count']} items")
                if "companies_found" in details:
                    print(f"         🏢 Companies: {details['companies_found']}")
                if "reading" in details:
                    print(f"         📖 Reading: {details['reading']}")
            
            if update.step == "completed":
                final_answer = details.get("final_answer", "")
                print("=" * 50)
                print("✅ RESEARCH COMPLETED!")
                print(f"📊 Total time: {elapsed:.1f}s")
                print(f"📝 Answer length: {len(final_answer)} characters")
                break
            elif update.step == "error":
                print(f"❌ ERROR: {update.message}")
                return False
        
        print("\n✅ Streaming test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_progress()
    sys.exit(0 if success else 1)

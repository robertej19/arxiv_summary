#!/usr/bin/env python3
"""
Test script to verify 10-K research agent setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import duckdb
        print("✅ duckdb")
    except ImportError as e:
        print(f"❌ duckdb: {e}")
        return False
    
    try:
        from smolagents import Tool
        print("✅ smolagents")
    except ImportError as e:
        print(f"❌ smolagents: {e}")
        return False
    
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python")
    except ImportError as e:
        print(f"❌ llama-cpp-python: {e}")
        return False
    
    return True

def test_database():
    """Test database connection."""
    print("\n📊 Testing 10-K database...")
    
    db_path = Path(__file__).parent.parent / "10k_knowledge_base.db"
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return False
    
    try:
        import duckdb
        conn = duckdb.connect(str(db_path))
        
        # Test basic query
        result = conn.execute("SELECT COUNT(*) FROM filings").fetchone()
        filing_count = result[0] if result else 0
        
        result = conn.execute("SELECT COUNT(*) FROM sections").fetchone()
        section_count = result[0] if result else 0
        
        conn.close()
        
        print(f"✅ Database connected successfully")
        print(f"   - {filing_count:,} filings")
        print(f"   - {section_count:,} sections")
        
        if filing_count == 0:
            print("⚠️  Warning: No filings found in database")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_model():
    """Test model availability."""
    print("\n🤖 Testing language model...")
    
    # Check model file exists
    model_paths = [
        Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        Path(__file__).parent.parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ Model file not found at expected locations:")
        for path in model_paths:
            print(f"   - {path}")
        print("\nTo download the model, run: bash download_model.sh")
        return False
    
    print(f"✅ Model found at {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024**3):.1f} GB")
    
    # Test loading (quick test)
    try:
        from llama_cpp_model import LlamaCppModel
        print("✅ Model wrapper imported successfully")
        return True
    except Exception as e:
        print(f"❌ Model wrapper error: {e}")
        return False

def test_tools():
    """Test 10-K tools."""
    print("\n🔧 Testing 10-K tools...")
    
    try:
        from tools_10k import TenKSearchTool, TenKReadTool
        
        # Test search tool initialization
        search_tool = TenKSearchTool()
        print("✅ TenKSearchTool initialized")
        
        # Test read tool initialization  
        read_tool = TenKReadTool()
        print("✅ TenKReadTool initialized")
        
        # Test a simple search (if database has data)
        try:
            result = search_tool.forward("business", limit=1)
            if result.get("results"):
                print(f"✅ Search test successful ({len(result['results'])} results)")
            else:
                print("⚠️  Search returned no results (database may be empty)")
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Tools error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 10-K Research Agent Setup Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database), 
        ("Model", test_model),
        ("Tools", test_tools),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! The 10-K research agent is ready to use.")
        print("\nTry running:")
        print("  python tenk_research_agent.py 'How do companies view AI risks?'")
        print("  python demo_10k_agent.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
        print("\nTo fix issues:")
        print("  - Install missing dependencies: pip install -r requirements_agent.txt")
        print("  - Build database: cd .. && python build_10k_knowledge_base.py")
        print("  - Download model: bash download_model.sh")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

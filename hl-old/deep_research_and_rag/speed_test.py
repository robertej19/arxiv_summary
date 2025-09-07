#!/usr/bin/env python3
"""
Speed Test & Comparison for 10-K Research Agents

Compare response times between:
1. Original deep research agent (minutes)
2. Fast agent (seconds) 
3. Ultra-fast agent (sub-second)

Usage:
    python speed_test.py
"""

import sys
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path

# Test questions of varying complexity
TEST_QUESTIONS = [
    # Simple factual questions (should be fast)
    "What are AI risks?",
    "How do companies view cybersecurity?", 
    "What supply chain issues do companies face?",
    
    # Medium complexity
    "How are technology companies approaching artificial intelligence investments?",
    "What climate change risks do manufacturing companies identify?",
    
    # Complex analytical questions  
    "Compare how different industries approach digital transformation and automation risks",
    "Analyze trends in regulatory compliance costs across sectors over the past 3 years",
]

def time_function(func, *args, **kwargs) -> Tuple[float, any]:
    """Time a function call and return (elapsed_time, result)."""
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return elapsed, result
    except Exception as e:
        elapsed = time.time() - start
        return elapsed, f"ERROR: {e}"


def test_ultra_fast_agent(question: str) -> Tuple[float, str]:
    """Test ultra-fast agent."""
    try:
        from ultra_fast_agent import ultra_fast_research
        return time_function(ultra_fast_research, question, verbose=False)
    except ImportError:
        return 0.0, "Ultra-fast agent not available"


def test_fast_agent(question: str) -> Tuple[float, str]:
    """Test fast agent."""
    try:
        from fast_tenk_agent import fast_10k_research  
        return time_function(fast_10k_research, question, verbose=False)
    except ImportError:
        return 0.0, "Fast agent not available"


def test_original_agent(question: str) -> Tuple[float, str]:
    """Test original deep research agent."""
    try:
        from tenk_research_agent import conduct_10k_research
        return time_function(conduct_10k_research, question)
    except ImportError:
        return 0.0, "Original agent not available"


def run_speed_benchmark():
    """Run comprehensive speed benchmark."""
    
    print("🚀 10-K RESEARCH AGENT SPEED BENCHMARK")
    print("=" * 60)
    print(f"Testing {len(TEST_QUESTIONS)} questions with 3 different agents")
    print("=" * 60)
    
    results = {
        "ultra_fast": [],
        "fast": [], 
        "original": []
    }
    
    agents = [
        ("Ultra-Fast", "ultra_fast", test_ultra_fast_agent),
        ("Fast", "fast", test_fast_agent),
        ("Original", "original", test_original_agent),
    ]
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 60)
        
        question_results = {}
        
        for agent_name, agent_key, agent_func in agents:
            print(f"🧪 Testing {agent_name} Agent...")
            
            elapsed, response = agent_func(question)
            results[agent_key].append(elapsed)
            question_results[agent_key] = {
                "time": elapsed,
                "response_length": len(str(response)) if response else 0,
                "success": not str(response).startswith("ERROR")
            }
            
            if elapsed > 0:
                print(f"   ⏱️  Time: {elapsed:.3f}s")
                print(f"   📝 Response: {len(str(response))} chars")
                if str(response).startswith("ERROR"):
                    print(f"   ❌ Error: {response}")
                else:
                    print(f"   ✅ Success")
            else:
                print(f"   ❌ Agent not available")
        
        # Show speed comparison for this question
        if any(question_results[key]["time"] > 0 for key in question_results):
            print(f"\n📊 Speed Comparison:")
            sorted_results = sorted(
                [(key, data) for key, data in question_results.items() if data["time"] > 0],
                key=lambda x: x[1]["time"]
            )
            
            for rank, (agent_key, data) in enumerate(sorted_results, 1):
                agent_names = {"ultra_fast": "Ultra-Fast", "fast": "Fast", "original": "Original"}
                speedup = sorted_results[0][1]["time"] / data["time"] if data["time"] > 0 else 1
                print(f"   {rank}. {agent_names[agent_key]}: {data['time']:.3f}s ({speedup:.1f}x faster)" if rank > 1 else f"   {rank}. {agent_names[agent_key]}: {data['time']:.3f}s (baseline)")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for agent_name, agent_key, _ in agents:
        times = [t for t in results[agent_key] if t > 0]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            success_rate = len(times) / len(TEST_QUESTIONS) * 100
            
            print(f"\n{agent_name} Agent:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Range: {min_time:.3f}s - {max_time:.3f}s")
            print(f"   Success Rate: {success_rate:.1f}%")
        else:
            print(f"\n{agent_name} Agent: Not available")
    
    # Speed improvement analysis
    ultra_times = [t for t in results["ultra_fast"] if t > 0]
    fast_times = [t for t in results["fast"] if t > 0]
    original_times = [t for t in results["original"] if t > 0]
    
    if ultra_times and original_times:
        ultra_avg = sum(ultra_times) / len(ultra_times)
        original_avg = sum(original_times) / len(original_times)
        speedup = original_avg / ultra_avg
        print(f"\n🚀 SPEEDUP ACHIEVED:")
        print(f"   Ultra-Fast vs Original: {speedup:.1f}x faster")
        print(f"   Time reduction: {original_avg - ultra_avg:.2f}s average")
    
    if fast_times and original_times:
        fast_avg = sum(fast_times) / len(fast_times)
        original_avg = sum(original_times) / len(original_times)
        speedup = original_avg / fast_avg
        print(f"   Fast vs Original: {speedup:.1f}x faster")
    
    return results


def quick_test():
    """Quick test with a single question."""
    
    question = "What are artificial intelligence risks?"
    
    print("🧪 QUICK SPEED TEST")
    print("=" * 40)
    print(f"Question: {question}")
    print("=" * 40)
    
    agents = [
        ("Ultra-Fast", test_ultra_fast_agent),
        ("Fast", test_fast_agent),
        ("Original", test_original_agent),
    ]
    
    for agent_name, agent_func in agents:
        print(f"\n🔬 Testing {agent_name} Agent...")
        elapsed, response = agent_func(question)
        
        if elapsed > 0:
            print(f"   ⏱️  Time: {elapsed:.3f}s")
            if not str(response).startswith("ERROR"):
                # Show first 200 chars of response
                preview = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
                print(f"   📝 Response preview: {preview}")
            else:
                print(f"   ❌ Error: {response}")
        else:
            print(f"   ❌ Agent not available")


def main():
    """CLI interface."""
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--quick', '-q', 'quick']:
        quick_test()
        return 0
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print("10-K Research Agent Speed Test")
        print("\nUsage:")
        print("  python speed_test.py           # Full benchmark")
        print("  python speed_test.py --quick   # Quick single test")
        print("  python speed_test.py --help    # Show this help")
        return 0
    
    try:
        results = run_speed_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ Benchmark complete! Check results above.")
        print("\nNext steps:")
        print("  - Run quick test: python speed_test.py --quick")
        print("  - Use ultra-fast: python ultra_fast_agent.py 'your question'")
        print("  - Use fast mode: python fast_tenk_agent.py 'your question'")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

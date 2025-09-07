#!/usr/bin/env python3
"""
Quick Speed Test - Test database queries without LLM overhead
"""

import time
import sys
from pathlib import Path
import duckdb

def test_database_speed():
    """Test raw database query speed."""
    print("🧪 TESTING DATABASE QUERY SPEED")
    print("=" * 50)
    
    # Find database
    db_path = Path("../10k_knowledge_base.db")
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "cybersecurity", 
        "supply chain",
        "climate change",
        "business risks"
    ]
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        
        start = time.time()
        conn = duckdb.connect(str(db_path))
        
        try:
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.fiscal_year,
                    s.section_name,
                    substr(s.content, 1, 200) as snippet
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE s.content ILIKE ?
                ORDER BY f.fiscal_year DESC
                LIMIT 3
            """
            
            results = conn.execute(sql, [f'%{query}%']).fetchall()
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   ⏱️  Query time: {elapsed:.3f}s")
            print(f"   📊 Results: {len(results)} items")
            
            if results:
                for ticker, company, year, section, snippet in results[:1]:
                    print(f"   📄 Sample: {company} ({ticker}) {year} - {section}")
            
        finally:
            conn.close()
    
    avg_time = total_time / len(test_queries)
    print(f"\n📊 SUMMARY:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average per query: {avg_time:.3f}s")
    print(f"   Queries per second: {1/avg_time:.1f}")


def test_ultra_fast_templates():
    """Test ultra-fast template responses."""
    print("\n🚀 TESTING ULTRA-FAST TEMPLATES")
    print("=" * 50)
    
    # Import ultra-fast agent
    sys.path.append('.')
    try:
        from ultra_fast_agent import ultra_fast_research
        
        test_questions = [
            "What are AI risks?",
            "How do companies approach cybersecurity?",
            "What supply chain issues do companies face?",
            "What climate change risks do companies identify?"
        ]
        
        total_time = 0
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: '{question}'")
            
            start = time.time()
            answer = ultra_fast_research(question, verbose=False)
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   ⏱️  Response time: {elapsed:.3f}s")
            print(f"   📝 Answer length: {len(answer)} chars")
            print(f"   ✅ Template used: {'Yes' if elapsed < 0.1 else 'No'}")
        
        avg_time = total_time / len(test_questions)
        print(f"\n📊 SUMMARY:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per question: {avg_time:.3f}s")
        print(f"   Speedup vs 30s baseline: {30/avg_time:.0f}x faster")
        
    except ImportError as e:
        print(f"❌ Could not import ultra_fast_agent: {e}")


def test_fast_search_only():
    """Test just the search part of fast mode (no LLM)."""
    print("\n⚡ TESTING FAST SEARCH (NO LLM)")
    print("=" * 50)
    
    sys.path.append('.')
    try:
        from fast_tenk_agent import FastDatabaseQuery, create_search_queries
        
        test_questions = [
            "How do companies approach AI investments?",
            "What cybersecurity risks do companies identify?",
            "How are supply chains being disrupted?",
            "What climate risks do manufacturing companies face?"
        ]
        
        db = FastDatabaseQuery()
        total_time = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: '{question}'")
            
            start = time.time()
            
            # Create search queries (fast)
            queries = create_search_queries(question)
            
            # Execute searches (this is what takes time)
            evidence_list = db.parallel_search(queries, limit_per_query=2)
            
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   🔍 Queries generated: {queries}")
            print(f"   ⏱️  Search time: {elapsed:.3f}s")
            print(f"   📊 Evidence found: {len(evidence_list)} items")
            
            if evidence_list:
                for ev in evidence_list[:1]:
                    print(f"   📄 Sample: {ev.company_name} ({ev.ticker}) {ev.fiscal_year}")
        
        avg_time = total_time / len(test_questions)
        print(f"\n📊 SUMMARY:")
        print(f"   Total search time: {total_time:.3f}s")
        print(f"   Average per question: {avg_time:.3f}s")
        print(f"   Search speed: {1/avg_time:.1f} questions/second")
        
    except ImportError as e:
        print(f"❌ Could not import fast search components: {e}")


def performance_comparison():
    """Show performance comparison table."""
    print("\n📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Mode           | Response Time | Use Case")
    print("-" * 60)
    print("Ultra-Fast     | < 0.1s       | Pre-computed templates")
    print("Fast (DB only) | 1-3s         | Custom questions (no LLM)")
    print("Fast (with LLM)| 30-45s       | Custom with AI synthesis")
    print("Deep Research  | 60-120s      | Comprehensive analysis")
    print("=" * 60)
    print("\n💡 KEY INSIGHTS:")
    print("• Ultra-fast templates are nearly instant")
    print("• Database queries are very fast (1-3s)")
    print("• LLM inference is the main bottleneck (30-40s)")
    print("• For speed, use templates or DB-only responses")


def main():
    """Run all speed tests."""
    print("🏁 COMPREHENSIVE SPEED TEST")
    print("=" * 60)
    
    # Test 1: Raw database speed
    test_database_speed()
    
    # Test 2: Ultra-fast templates
    test_ultra_fast_templates()
    
    # Test 3: Fast search without LLM
    test_fast_search_only()
    
    # Show comparison
    performance_comparison()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Use ultra-fast mode for common topics (AI, cybersecurity, etc.)")
    print("2. For custom questions, consider DB-only responses")
    print("3. Use LLM synthesis only when necessary")
    print("4. The database itself is very fast - LLM is the bottleneck")


if __name__ == "__main__":
    main()

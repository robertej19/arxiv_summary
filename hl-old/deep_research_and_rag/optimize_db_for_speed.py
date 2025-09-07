#!/usr/bin/env python3
"""
Database Speed Optimization for 10-K Knowledge Base

Creates additional indexes and materializes views for ultra-fast queries.
Run this once after building your knowledge base.

Usage:
    python optimize_db_for_speed.py
"""

import sys
import time
from pathlib import Path
import duckdb


def optimize_database(db_path: str = "10k_knowledge_base.db"):
    """Add speed optimizations to existing database."""
    
    db_file = Path(db_path)
    if not db_file.exists():
        # Try parent directory
        parent_db = Path(__file__).parent.parent / db_path
        if parent_db.exists():
            db_file = parent_db
        else:
            print(f"❌ Database not found: {db_path}")
            return False
    
    print(f"🔧 Optimizing database: {db_file}")
    
    conn = duckdb.connect(str(db_file))
    
    try:
        # 1. Create full-text search index (if not exists)
        print("📇 Creating content search indexes...")
        
        # Index on content for faster LIKE queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sections_content_trgm 
            ON sections USING gin(content gin_trgm_ops)
        """)
        
        # Index on section content length for efficient filtering
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sections_content_length 
            ON sections(content_length)
        """)
        
        # 2. Create materialized views for common queries
        print("📊 Creating materialized views...")
        
        # Company summary view
        conn.execute("""
            CREATE OR REPLACE VIEW v_company_summary AS
            SELECT 
                f.ticker,
                f.company_name,
                COUNT(DISTINCT f.fiscal_year) as year_count,
                MIN(f.fiscal_year) as earliest_year,
                MAX(f.fiscal_year) as latest_year,
                COUNT(s.id) as section_count,
                SUM(s.content_length) as total_content_length
            FROM filings f
            LEFT JOIN sections s ON f.id = s.filing_id
            GROUP BY f.ticker, f.company_name
        """)
        
        # Recent filings view (last 3 years)
        conn.execute("""
            CREATE OR REPLACE VIEW v_recent_filings AS
            SELECT 
                f.ticker,
                f.company_name,
                f.fiscal_year,
                f.filing_date,
                s.section_name,
                s.content_length,
                s.word_count,
                -- Pre-compute common snippets
                substr(s.content, 1, 500) as preview_snippet
            FROM filings f
            JOIN sections s ON f.id = s.filing_id
            WHERE f.fiscal_year >= (
                SELECT MAX(fiscal_year) - 2 FROM filings
            )
        """)
        
        # 3. Create keyword search optimization table
        print("🔍 Creating keyword optimization table...")
        
        # Create table for fast keyword lookup
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keyword_cache (
                keyword VARCHAR,
                ticker VARCHAR,
                company_name VARCHAR,
                fiscal_year INTEGER,
                section_name VARCHAR,
                snippet TEXT,
                relevance_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (keyword, ticker, fiscal_year, section_name)
            )
        """)
        
        # Index for keyword cache
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_keyword_cache_keyword 
            ON keyword_cache(keyword)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_keyword_cache_score 
            ON keyword_cache(keyword, relevance_score DESC)
        """)
        
        # 4. Pre-populate common keywords
        print("💾 Pre-populating keyword cache...")
        
        common_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'automation',
            'cybersecurity', 'cyber security', 'data breach', 'privacy',
            'supply chain', 'logistics', 'shortage', 'disruption',
            'climate change', 'environmental', 'carbon', 'sustainability',
            'regulation', 'compliance', 'risk', 'competition',
            'revenue', 'costs', 'expenses', 'profit', 'growth'
        ]
        
        for keyword in common_keywords:
            # Clear existing entries for this keyword
            conn.execute("DELETE FROM keyword_cache WHERE keyword = ?", [keyword])
            
            # Insert top results for this keyword
            conn.execute("""
                INSERT INTO keyword_cache 
                (keyword, ticker, company_name, fiscal_year, section_name, snippet, relevance_score)
                SELECT 
                    ? as keyword,
                    f.ticker,
                    f.company_name,
                    f.fiscal_year,
                    s.section_name,
                    CASE 
                        WHEN position(lower(?) in lower(s.content)) > 0 THEN
                            substr(s.content, 
                                greatest(1, position(lower(?) in lower(s.content)) - 100),
                                400)
                        ELSE
                            substr(s.content, 1, 400)
                    END as snippet,
                    -- Simple relevance scoring
                    CASE 
                        WHEN s.section_name ILIKE '%business%' THEN 1.0
                        WHEN s.section_name ILIKE '%risk%' THEN 0.9  
                        ELSE 0.8
                    END as relevance_score
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE s.content ILIKE ?
                ORDER BY f.fiscal_year DESC, relevance_score DESC
                LIMIT 10
            """, [keyword, keyword, keyword, f'%{keyword}%'])
        
        # 5. Create statistics table for ultra-fast responses
        print("📈 Creating statistics cache...")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats_cache (
                stat_name VARCHAR PRIMARY KEY,
                stat_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Populate basic stats
        stats_queries = {
            'total_companies': "SELECT COUNT(DISTINCT ticker) FROM filings",
            'total_filings': "SELECT COUNT(*) FROM filings", 
            'year_range': "SELECT MIN(fiscal_year) || '-' || MAX(fiscal_year) FROM filings",
            'total_sections': "SELECT COUNT(*) FROM sections",
            'latest_year': "SELECT MAX(fiscal_year) FROM filings"
        }
        
        for stat_name, query in stats_queries.items():
            result = conn.execute(query).fetchone()
            stat_value = str(result[0]) if result else "0"
            
            conn.execute("""
                INSERT OR REPLACE INTO stats_cache (stat_name, stat_value)
                VALUES (?, ?)
            """, [stat_name, stat_value])
        
        # 6. Analyze tables for query optimization
        print("🔍 Analyzing tables for query optimization...")
        
        conn.execute("ANALYZE filings")
        conn.execute("ANALYZE sections") 
        conn.execute("ANALYZE keyword_cache")
        
        # 7. Test optimization results
        print("🧪 Testing optimization performance...")
        
        start_time = time.time()
        result = conn.execute("""
            SELECT COUNT(*) FROM keyword_cache WHERE keyword = 'ai'
        """).fetchone()
        cache_time = time.time() - start_time
        
        start_time = time.time()
        result = conn.execute("""
            SELECT ticker, company_name FROM v_recent_filings 
            WHERE section_name ILIKE '%business%' LIMIT 5
        """).fetchall()
        view_time = time.time() - start_time
        
        print(f"✅ Optimization complete!")
        print(f"   - Keyword cache query: {cache_time*1000:.1f}ms")
        print(f"   - Recent filings view: {view_time*1000:.1f}ms")
        print(f"   - Cached {len(common_keywords)} common keywords")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False
        
    finally:
        conn.close()


def show_optimization_stats(db_path: str = "10k_knowledge_base.db"):
    """Show current optimization status."""
    
    db_file = Path(db_path)
    if not db_file.exists():
        parent_db = Path(__file__).parent.parent / db_path
        if parent_db.exists():
            db_file = parent_db
        else:
            print(f"❌ Database not found: {db_path}")
            return
    
    conn = duckdb.connect(str(db_file))
    
    try:
        print("📊 DATABASE OPTIMIZATION STATUS")
        print("=" * 40)
        
        # Check if keyword cache exists
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = 'keyword_cache'
        """).fetchall()
        
        if tables:
            # Show keyword cache stats
            result = conn.execute("SELECT COUNT(DISTINCT keyword) FROM keyword_cache").fetchone()
            keyword_count = result[0] if result else 0
            
            result = conn.execute("SELECT COUNT(*) FROM keyword_cache").fetchone()
            total_cached = result[0] if result else 0
            
            print(f"✅ Keyword cache: {keyword_count} keywords, {total_cached} entries")
            
            # Show top keywords
            results = conn.execute("""
                SELECT keyword, COUNT(*) as count 
                FROM keyword_cache 
                GROUP BY keyword 
                ORDER BY count DESC 
                LIMIT 5
            """).fetchall()
            
            print("   Top cached keywords:")
            for keyword, count in results:
                print(f"     - {keyword}: {count} entries")
        else:
            print("❌ Keyword cache not found")
        
        # Check views
        views = conn.execute("""
            SELECT table_name FROM information_schema.views 
            WHERE table_name IN ('v_company_summary', 'v_recent_filings')
        """).fetchall()
        
        print(f"✅ Optimized views: {len(views)}/2 created")
        
        # Check stats cache
        stats_tables = conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = 'stats_cache'
        """).fetchall()
        
        if stats_tables:
            result = conn.execute("SELECT COUNT(*) FROM stats_cache").fetchone()
            stats_count = result[0] if result else 0
            print(f"✅ Statistics cache: {stats_count} cached stats")
        else:
            print("❌ Statistics cache not found")
        
    finally:
        conn.close()


def main():
    """CLI interface."""
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--status', '-s', 'status']:
        show_optimization_stats()
        return 0
    
    print("🚀 10-K DATABASE SPEED OPTIMIZATION")
    print("=" * 40)
    print("This will add indexes and caches for faster queries.")
    print()
    
    # Check if database exists
    db_path = "10k_knowledge_base.db"
    if not Path(db_path).exists():
        parent_db = Path(__file__).parent.parent / db_path
        if parent_db.exists():
            db_path = str(parent_db)
        else:
            print("❌ Database not found. Please build the knowledge base first:")
            print("   cd .. && python build_10k_knowledge_base.py")
            return 1
    
    try:
        success = optimize_database(db_path)
        
        if success:
            print("\n🎉 Database optimization complete!")
            print("\nNext steps:")
            print("  - Test fast agent: python fast_tenk_agent.py 'your question'")
            print("  - Test ultra-fast: python ultra_fast_agent.py 'your question'")
            print("  - Check status: python optimize_db_for_speed.py --status")
            return 0
        else:
            print("\n❌ Optimization failed. Check error messages above.")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

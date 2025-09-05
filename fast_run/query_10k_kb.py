#!/usr/bin/env python3
"""
Query interface for the 10-K Knowledge Base.

Provides fast querying capabilities for research and RAG applications.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import duckdb
import pandas as pd

class TenKQueryInterface:
    """Fast query interface for 10-K knowledge base."""
    
    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
    def search_filings(self, 
                      query: str,
                      tickers: Optional[List[str]] = None,
                      years: Optional[List[int]] = None,
                      sections: Optional[List[str]] = None,
                      limit: int = 10) -> pd.DataFrame:
        """Advanced search across all filings."""
        
        sql = """
            SELECT 
                f.ticker,
                f.company_name,
                f.filing_date,
                f.fiscal_year,
                s.section_name,
                LENGTH(s.content) as content_length,
                s.content
            FROM sections s
            JOIN filings f ON s.filing_id = f.id
            WHERE s.content ILIKE ?
        """
        
        params = [f'%{query}%']
        
        if tickers:
            placeholders = ','.join(['?' for _ in tickers])
            sql += f" AND f.ticker IN ({placeholders})"
            params.extend(tickers)
            
        if years:
            placeholders = ','.join(['?' for _ in years])
            sql += f" AND f.fiscal_year IN ({placeholders})"
            params.extend(years)
            
        if sections:
            placeholders = ','.join(['?' for _ in sections])
            sql += f" AND s.section_name IN ({placeholders})"
            params.extend(sections)
            
        sql += " ORDER BY f.filing_date DESC, s.section_name LIMIT ?"
        params.append(limit)
        
        return self.conn.execute(sql, params).df()
        
    def get_company_overview(self, ticker: str) -> Dict:
        """Get comprehensive overview of a company's filings."""
        
        # Basic company info
        company_info = self.conn.execute("""
            SELECT 
                ticker,
                company_name,
                COUNT(*) as total_filings,
                MIN(filing_date) as earliest_filing,
                MAX(filing_date) as latest_filing,
                AVG(full_text_length) as avg_filing_length
            FROM filings 
            WHERE ticker = ?
            GROUP BY ticker, company_name
        """, (ticker,)).fetchone()
        
        if not company_info:
            return {"error": f"No filings found for ticker {ticker}"}
            
        # Filing details by year
        yearly_filings = self.conn.execute("""
            SELECT fiscal_year, COUNT(*) as count
            FROM filings 
            WHERE ticker = ?
            GROUP BY fiscal_year
            ORDER BY fiscal_year DESC
        """, (ticker,)).fetchall()
        
        # Section availability
        section_availability = self.conn.execute("""
            SELECT s.section_name, COUNT(*) as count
            FROM sections s
            JOIN filings f ON s.filing_id = f.id
            WHERE f.ticker = ?
            GROUP BY s.section_name
            ORDER BY count DESC
        """, (ticker,)).fetchall()
        
        return {
            "ticker": company_info[0],
            "company_name": company_info[1],
            "total_filings": company_info[2],
            "filing_date_range": (company_info[3], company_info[4]),
            "avg_filing_length": company_info[5],
            "yearly_filings": dict(yearly_filings),
            "section_availability": dict(section_availability)
        }
        
    def compare_companies(self, tickers: List[str], metric: str = "risk_factors") -> pd.DataFrame:
        """Compare specific aspects across companies."""
        
        section_map = {
            "business": "section_item_1",
            "risk_factors": "section_item_1a", 
            "md_a": "section_item_7",
            "financial_statements": "section_item_8"
        }
        
        section_name = section_map.get(metric, f"section_{metric}")
        
        sql = """
            SELECT 
                f.ticker,
                f.company_name,
                f.fiscal_year,
                f.filing_date,
                s.word_count,
                s.content_length
            FROM sections s
            JOIN filings f ON s.filing_id = f.id
            WHERE f.ticker = ANY(?) AND s.section_name = ?
            ORDER BY f.ticker, f.fiscal_year DESC
        """
        
        return self.conn.execute(sql, (tickers, section_name)).df()
        
    def get_trending_topics(self, 
                           section: str = "section_item_1a",
                           min_year: int = 2020,
                           top_n: int = 20) -> pd.DataFrame:
        """Find trending topics/keywords in specific sections."""
        
        # This is a simplified version - in practice you'd want proper NLP
        sql = """
            WITH word_counts AS (
                SELECT 
                    f.fiscal_year,
                    f.ticker,
                    LENGTH(s.content) - LENGTH(REPLACE(LOWER(s.content), ?, '')) as word_freq,
                    s.content_length
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE s.section_name = ? AND f.fiscal_year >= ?
            )
            SELECT 
                fiscal_year,
                COUNT(*) as companies_mentioning,
                AVG(word_freq::FLOAT / content_length * 1000) as avg_frequency_per_1000_chars
            FROM word_counts
            WHERE word_freq > 0
            GROUP BY fiscal_year
            ORDER BY fiscal_year DESC
        """
        
        # Example with some key terms
        terms = ['artificial intelligence', 'climate', 'pandemic', 'supply chain', 
                'cybersecurity', 'inflation', 'recession', 'technology']
        
        results = []
        for term in terms:
            term_data = self.conn.execute(sql, (term, section, min_year)).fetchall()
            for year, companies, freq in term_data:
                results.append({
                    'term': term,
                    'year': year,
                    'companies_mentioning': companies,
                    'avg_frequency': freq
                })
                
        return pd.DataFrame(results)
        
    def export_for_rag(self, 
                      output_dir: str = "rag_chunks",
                      chunk_size: int = 1000,
                      overlap: int = 200) -> None:
        """Export data in chunks suitable for RAG systems."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all sections
        sections = self.conn.execute("""
            SELECT 
                f.ticker,
                f.company_name,
                f.fiscal_year,
                f.filing_date,
                s.section_name,
                s.content,
                f.file_path
            FROM sections s
            JOIN filings f ON s.filing_id = f.id
            WHERE s.content_length > 500
            ORDER BY f.ticker, f.fiscal_year DESC, s.section_name
        """).fetchall()
        
        chunks = []
        chunk_id = 0
        
        for ticker, company, year, filing_date, section, content, file_path in sections:
            # Split content into chunks
            words = content.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) < 100:  # Skip very short chunks
                    continue
                    
                chunks.append({
                    'chunk_id': chunk_id,
                    'ticker': ticker,
                    'company_name': company,
                    'fiscal_year': year,
                    'filing_date': filing_date,
                    'section_name': section,
                    'chunk_text': chunk_text,
                    'source_file': file_path,
                    'chunk_start': i,
                    'chunk_end': min(i + chunk_size, len(words))
                })
                
                chunk_id += 1
                
        # Save as parquet for efficient loading
        chunks_df = pd.DataFrame(chunks)
        chunks_df.to_parquet(output_path / "rag_chunks.parquet", compression='snappy')
        
        # Also save metadata
        metadata = {
            'total_chunks': len(chunks),
            'chunk_size': chunk_size,
            'overlap': overlap,
            'companies': list(chunks_df['ticker'].unique()),
            'sections': list(chunks_df['section_name'].unique()),
            'date_range': [str(chunks_df['filing_date'].min()), str(chunks_df['filing_date'].max())]
        }
        
        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Exported {len(chunks)} chunks to {output_path}")
        
    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Query 10-K Knowledge Base")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--ticker", help="Company ticker")
    parser.add_argument("--tickers", nargs="+", help="Multiple company tickers")
    parser.add_argument("--years", nargs="+", type=int, help="Fiscal years")
    parser.add_argument("--sections", nargs="+", help="Section names")
    parser.add_argument("--overview", action="store_true", help="Company overview")
    parser.add_argument("--compare", help="Compare companies on metric")
    parser.add_argument("--trending", action="store_true", help="Find trending topics")
    parser.add_argument("--export-rag", action="store_true", help="Export for RAG")
    parser.add_argument("--limit", type=int, default=10, help="Result limit")
    parser.add_argument("--db", default="10k_knowledge_base.db", help="Database path")
    
    args = parser.parse_args()
    
    if not Path(args.db).exists():
        print(f"Database {args.db} not found. Run build_10k_knowledge_base.py first.")
        return
        
    query_interface = TenKQueryInterface(args.db)
    
    try:
        if args.search:
            results = query_interface.search_filings(
                query=args.search,
                tickers=args.tickers,
                years=args.years,
                sections=args.sections,
                limit=args.limit
            )
            print(f"\nFound {len(results)} results:")
            for _, row in results.iterrows():
                print(f"\n{row['ticker']} ({row['fiscal_year']}) - {row['section_name']}")
                preview = row['content'][:300] + "..." if len(row['content']) > 300 else row['content']
                print(f"  {preview}")
                
        elif args.overview and args.ticker:
            overview = query_interface.get_company_overview(args.ticker)
            print(f"\nCompany Overview: {overview['ticker']}")
            print("-" * 40)
            for key, value in overview.items():
                if key != 'ticker':
                    print(f"{key}: {value}")
                    
        elif args.compare and args.tickers:
            comparison = query_interface.compare_companies(args.tickers, args.compare)
            print(f"\nComparing {args.compare} across companies:")
            print(comparison.to_string(index=False))
            
        elif args.trending:
            trends = query_interface.get_trending_topics()
            print("\nTrending topics in Risk Factors:")
            print(trends.to_string(index=False))
            
        elif args.export_rag:
            query_interface.export_for_rag()
            
        else:
            parser.print_help()
            
    finally:
        query_interface.close()


if __name__ == "__main__":
    main()

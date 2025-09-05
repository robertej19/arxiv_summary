#!/usr/bin/env python3
"""
Build an ultrafast knowledge base from 10-K filings using DuckDB and Parquet.

This script:
1. Parses HTML 10-K filings into structured data
2. Extracts key sections (Business, Risk Factors, MD&A, etc.)
3. Stores everything in DuckDB with Parquet backing for ultra-fast queries
4. Creates optimized indexes for research and RAG applications

Requires: beautifulsoup4, duckdb, pandas, pyarrow
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import logging

import duckdb
import pandas as pd
from bs4 import BeautifulSoup, Comment
import pyarrow as pa
import pyarrow.parquet as pq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TenKParser:
    """Parser for SEC 10-K HTML filings."""
    
    # Standard 10-K sections we want to extract
    SECTIONS = {
        'item_1': r'item\s*1[^0-9a-z].*?business',
        'item_1a': r'item\s*1a[^0-9a-z].*?risk\s*factors',
        'item_2': r'item\s*2[^0-9a-z].*?properties',
        'item_3': r'item\s*3[^0-9a-z].*?legal\s*proceedings',
        'item_4': r'item\s*4[^0-9a-z].*?mine\s*safety',
        'item_5': r'item\s*5[^0-9a-z].*?market\s*for.*common\s*equity',
        'item_6': r'item\s*6[^0-9a-z].*?financial\s*data',
        'item_7': r'item\s*7[^0-9a-z].*?management.*discussion.*analysis',
        'item_7a': r'item\s*7a[^0-9a-z].*?market\s*risk',
        'item_8': r'item\s*8[^0-9a-z].*?financial\s*statements',
        'item_9': r'item\s*9[^0-9a-z].*?controls\s*and\s*procedures',
        'item_9a': r'item\s*9a[^0-9a-z].*?controls\s*and\s*procedures',
        'item_9b': r'item\s*9b[^0-9a-z].*?other\s*information',
        'item_10': r'item\s*10[^0-9a-z].*?directors.*officers',
        'item_11': r'item\s*11[^0-9a-z].*?compensation',
        'item_12': r'item\s*12[^0-9a-z].*?security\s*ownership',
        'item_13': r'item\s*13[^0-9a-z].*?relationships.*transactions',
        'item_14': r'item\s*14[^0-9a-z].*?principal\s*accountant',
        'item_15': r'item\s*15[^0-9a-z].*?exhibits',
        'part_i': r'part\s*i[^a-z]',
        'part_ii': r'part\s*ii[^a-z]',
        'part_iii': r'part\s*iii[^a-z]',
        'part_iv': r'part\s*iv[^a-z]',
    }
    
    def __init__(self):
        self.soup = None
        self.text = None
        
    def parse_file(self, file_path: Path) -> Dict:
        """Parse a 10-K HTML file and extract structured data."""
        logger.info(f"Parsing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
            
        # Parse with BeautifulSoup
        self.soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in self.soup(["script", "style"]):
            script.decompose()
            
        # Remove comments
        for comment in self.soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
            
        # Get clean text
        self.text = self.soup.get_text()
        
        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(file_path)
        
        # Extract document metadata
        doc_metadata = self._extract_document_metadata()
        metadata.update(doc_metadata)
        
        # Extract sections
        sections = self._extract_sections()
        
        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        result = {
            'file_path': str(file_path),
            'content_hash': content_hash,
            'parsed_date': datetime.now().isoformat(),
            'file_size_bytes': len(content),
            **metadata,
            **sections
        }
        
        return result
        
    def _extract_metadata_from_filename(self, file_path: Path) -> Dict:
        """Extract metadata from the filename pattern: YYYY-MM-DD_TICKER_10-K_filename.htm"""
        filename = file_path.name
        parts = filename.split('_')
        
        metadata = {
            'ticker': None,
            'filing_date': None,
            'fiscal_year': None,
        }
        
        if len(parts) >= 3:
            try:
                filing_date = datetime.strptime(parts[0], '%Y-%m-%d').date()
                metadata['filing_date'] = filing_date.isoformat()
                metadata['fiscal_year'] = filing_date.year
            except ValueError:
                pass
                
            metadata['ticker'] = parts[1] if len(parts) > 1 else None
            
        # Extract from parent directory (ticker)
        if not metadata['ticker']:
            metadata['ticker'] = file_path.parent.name
            
        return metadata
        
    def _extract_document_metadata(self) -> Dict:
        """Extract metadata from the document itself."""
        metadata = {
            'company_name': None,
            'cik': None,
            'sic_code': None,
            'state_of_incorporation': None,
            'fiscal_year_end': None,
        }
        
        # Try to find company name in title or early content
        title = self.soup.find('title')
        if title:
            title_text = title.get_text().strip()
            metadata['document_title'] = title_text
            
        # Look for SEC metadata in XBRL tags
        for tag in self.soup.find_all(['ix:nonnumeric', 'ix:nonfraction']):
            name = tag.get('name', '').lower()
            if 'entityregistrantname' in name:
                metadata['company_name'] = tag.get_text().strip()
            elif 'entitycentralindexkey' in name:
                metadata['cik'] = tag.get_text().strip()
            elif 'entityincorporationstatecountrycode' in name:
                metadata['state_of_incorporation'] = tag.get_text().strip()
                
        # Try to extract from text patterns
        if not metadata['company_name']:
            # Look for company name patterns
            patterns = [
                r'(?:UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?)(.*?)(?:\n.*?Form 10-K)',
                r'Form 10-K.*?\n.*?\n(.*?)(?:\n|$)',
            ]
            for pattern in patterns:
                match = re.search(pattern, self.text[:5000], re.IGNORECASE | re.DOTALL)
                if match:
                    potential_name = match.group(1).strip()
                    if len(potential_name) < 100 and potential_name:
                        metadata['company_name'] = potential_name
                        break
                        
        return metadata
        
    def _extract_sections(self) -> Dict:
        """Extract standard 10-K sections."""
        sections = {}
        text_lower = self.text.lower()
        
        # Find section boundaries
        section_positions = {}
        
        for section_key, pattern in self.SECTIONS.items():
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                # Take the first match
                section_positions[section_key] = matches[0].start()
                
        # Sort sections by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
        
        # Extract content between sections
        for i, (section_key, start_pos) in enumerate(sorted_sections):
            # Determine end position
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = len(self.text)
                
            # Extract section content
            section_content = self.text[start_pos:end_pos].strip()
            
            # Clean up the content
            section_content = self._clean_section_content(section_content)
            
            if section_content and len(section_content) > 100:  # Only include substantial sections
                sections[f'section_{section_key}'] = section_content
                sections[f'section_{section_key}_length'] = len(section_content)
                sections[f'section_{section_key}_word_count'] = len(section_content.split())
                
        # Add full text as well
        sections['full_text'] = self.text
        sections['full_text_length'] = len(self.text)
        sections['full_text_word_count'] = len(self.text.split())
        
        return sections
        
    def _clean_section_content(self, content: str) -> str:
        """Clean section content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers and headers/footers
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
        content = re.sub(r'\n\s*Page \d+.*?\n', '\n', content, flags=re.IGNORECASE)
        
        # Remove table of contents references
        content = re.sub(r'\.{3,}\d+', '', content)
        
        return content.strip()


class TenKKnowledgeBase:
    """Ultra-fast knowledge base for 10-K filings using DuckDB and Parquet."""
    
    def __init__(self, db_path: str = "10k_knowledge_base.db", parquet_dir: str = "parquet_data"):
        self.db_path = db_path
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(exist_ok=True)
        
        self.parser = TenKParser()
        
        # Initialize DuckDB
        self.conn = duckdb.connect(self.db_path)
        self._setup_database()
        
    def _setup_database(self):
        """Set up DuckDB database schema."""
        
        # Drop existing tables to fix schema issues
        self.conn.execute("DROP TABLE IF EXISTS sections")
        self.conn.execute("DROP TABLE IF EXISTS filings")
        
        # Main filings table with proper auto-increment
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS filing_id_seq
        """)
        
        self.conn.execute("""
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY DEFAULT nextval('filing_id_seq'),
                file_path VARCHAR,
                content_hash VARCHAR UNIQUE,
                parsed_date TIMESTAMP,
                file_size_bytes BIGINT,
                ticker VARCHAR,
                filing_date DATE,
                fiscal_year INTEGER,
                company_name VARCHAR,
                cik VARCHAR,
                sic_code VARCHAR,
                state_of_incorporation VARCHAR,
                fiscal_year_end VARCHAR,
                document_title VARCHAR,
                full_text_length BIGINT,
                full_text_word_count BIGINT
            )
        """)
        
        # Sections table for detailed section analysis
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS section_id_seq
        """)
        
        self.conn.execute("""
            CREATE TABLE sections (
                id INTEGER PRIMARY KEY DEFAULT nextval('section_id_seq'),
                filing_id INTEGER,
                section_name VARCHAR,
                content TEXT,
                content_length BIGINT,
                word_count BIGINT,
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)
        
        # Create indexes for fast queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filings_fiscal_year ON filings(fiscal_year)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filings_filing_date ON filings(filing_date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_filing_id ON sections(filing_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_section_name ON sections(section_name)")
        
        logger.info("Database schema initialized")
        
    def ingest_filings(self, filings_dir: Path):
        """Ingest all 10-K filings from the specified directory."""
        
        filing_files = list(filings_dir.glob("**/*.htm*"))
        logger.info(f"Found {len(filing_files)} filing files to process")
        
        processed_count = 0
        skipped_count = 0
        
        for file_path in filing_files:
            try:
                # Parse the filing
                parsed_data = self.parser.parse_file(file_path)
                
                if not parsed_data:
                    logger.warning(f"Failed to parse {file_path}")
                    continue
                    
                # Check if already processed (by content hash)
                existing = self.conn.execute(
                    "SELECT id FROM filings WHERE content_hash = ?", 
                    (parsed_data['content_hash'],)
                ).fetchone()
                
                if existing:
                    logger.info(f"Skipping {file_path} - already processed")
                    skipped_count += 1
                    continue
                    
                # Insert into database
                self._insert_filing(parsed_data)
                processed_count += 1
                
                if processed_count % 5 == 0:
                    logger.info(f"Processed {processed_count} filings...")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
                
        logger.info(f"Ingestion complete: {processed_count} processed, {skipped_count} skipped")
        
        # Export to Parquet for ultra-fast analytics
        self._export_to_parquet()
        
    def _insert_filing(self, data: Dict):
        """Insert parsed filing data into the database."""
        
        # Extract sections from data
        sections_data = {}
        main_data = {}
        
        for key, value in data.items():
            if key.startswith('section_') and not key.endswith('_length') and not key.endswith('_word_count'):
                sections_data[key] = {
                    'content': value,
                    'length': data.get(f'{key}_length', 0),
                    'word_count': data.get(f'{key}_word_count', 0)
                }
            elif not key.startswith('section_') or key in ['full_text_length', 'full_text_word_count']:
                main_data[key] = value
                
        # Remove full_text from main_data to keep it lean
        full_text = main_data.pop('full_text', '')
        
        # Insert main filing record - let DuckDB auto-generate the ID
        filing_id = self.conn.execute("""
            INSERT INTO filings (
                file_path, content_hash, parsed_date, file_size_bytes,
                ticker, filing_date, fiscal_year, company_name, cik,
                sic_code, state_of_incorporation, fiscal_year_end,
                document_title, full_text_length, full_text_word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """, (
            main_data.get('file_path'),
            main_data.get('content_hash'),
            main_data.get('parsed_date'),
            main_data.get('file_size_bytes'),
            main_data.get('ticker'),
            main_data.get('filing_date'),
            main_data.get('fiscal_year'),
            main_data.get('company_name'),
            main_data.get('cik'),
            main_data.get('sic_code'),
            main_data.get('state_of_incorporation'),
            main_data.get('fiscal_year_end'),
            main_data.get('document_title'),
            main_data.get('full_text_length'),
            main_data.get('full_text_word_count'),
        )).fetchone()[0]
        
        # Insert sections - let DuckDB auto-generate the section IDs
        for section_name, section_info in sections_data.items():
            self.conn.execute("""
                INSERT INTO sections (filing_id, section_name, content, content_length, word_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                filing_id,
                section_name,
                section_info['content'],
                section_info['length'],
                section_info['word_count']
            ))
            
        # Also store full text as a section
        if full_text:
            self.conn.execute("""
                INSERT INTO sections (filing_id, section_name, content, content_length, word_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                filing_id,
                'full_text',
                full_text,
                len(full_text),
                len(full_text.split())
            ))
            
    def _export_to_parquet(self):
        """Export data to Parquet files for ultra-fast analytics."""
        
        logger.info("Exporting to Parquet format...")
        
        # Export filings table
        filings_df = self.conn.execute("SELECT * FROM filings").df()
        filings_parquet_path = self.parquet_dir / "filings.parquet"
        filings_df.to_parquet(filings_parquet_path, engine='pyarrow', compression='snappy')
        
        # Export sections table (in chunks by section type for better query performance)
        section_types = self.conn.execute("SELECT DISTINCT section_name FROM sections").fetchall()
        
        for (section_name,) in section_types:
            section_df = self.conn.execute(
                "SELECT * FROM sections WHERE section_name = ?", 
                (section_name,)
            ).df()
            
            section_parquet_path = self.parquet_dir / f"sections_{section_name}.parquet"
            section_df.to_parquet(section_parquet_path, engine='pyarrow', compression='snappy')
            
        logger.info(f"Parquet files exported to {self.parquet_dir}")
        
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        
        stats = {}
        
        # Filing stats
        stats['total_filings'] = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        stats['unique_companies'] = self.conn.execute("SELECT COUNT(DISTINCT ticker) FROM filings").fetchone()[0]
        stats['date_range'] = self.conn.execute(
            "SELECT MIN(filing_date), MAX(filing_date) FROM filings WHERE filing_date IS NOT NULL"
        ).fetchone()
        
        # Content stats
        stats['total_text_length'] = self.conn.execute("SELECT SUM(full_text_length) FROM filings").fetchone()[0]
        stats['average_filing_length'] = self.conn.execute("SELECT AVG(full_text_length) FROM filings").fetchone()[0]
        
        # Section stats
        stats['total_sections'] = self.conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        stats['sections_by_type'] = dict(self.conn.execute(
            "SELECT section_name, COUNT(*) FROM sections GROUP BY section_name ORDER BY COUNT(*) DESC"
        ).fetchall())
        
        # Company stats
        stats['filings_by_company'] = dict(self.conn.execute(
            "SELECT ticker, COUNT(*) FROM filings WHERE ticker IS NOT NULL GROUP BY ticker ORDER BY COUNT(*) DESC"
        ).fetchall())
        
        return stats
        
    def search_content(self, query: str, section_filter: Optional[str] = None, 
                      ticker_filter: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Search content across all filings."""
        
        sql = """
            SELECT f.ticker, f.company_name, f.filing_date, f.fiscal_year,
                   s.section_name, s.content, f.file_path
            FROM sections s
            JOIN filings f ON s.filing_id = f.id
            WHERE s.content LIKE ?
        """
        
        params = [f'%{query}%']
        
        if section_filter:
            sql += " AND s.section_name = ?"
            params.append(section_filter)
            
        if ticker_filter:
            sql += " AND f.ticker = ?"
            params.append(ticker_filter)
            
        sql += " ORDER BY f.filing_date DESC LIMIT ?"
        params.append(limit)
        
        results = self.conn.execute(sql, params).fetchall()
        
        columns = ['ticker', 'company_name', 'filing_date', 'fiscal_year', 
                  'section_name', 'content', 'file_path']
        
        return [dict(zip(columns, row)) for row in results]
        
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main function to build the knowledge base."""
    
    # Configuration
    filings_dir = Path("10k_filings")
    db_path = "10k_knowledge_base.db"
    parquet_dir = "parquet_data"
    
    if not filings_dir.exists():
        logger.error(f"Filings directory {filings_dir} not found!")
        return
        
    # Initialize knowledge base
    kb = TenKKnowledgeBase(db_path=db_path, parquet_dir=parquet_dir)
    
    try:
        # Ingest all filings
        kb.ingest_filings(filings_dir)
        
        # Print statistics
        stats = kb.get_stats()
        print("\n" + "="*60)
        print("10-K KNOWLEDGE BASE STATISTICS")
        print("="*60)
        print(f"Total filings: {stats['total_filings']:,}")
        print(f"Unique companies: {stats['unique_companies']:,}")
        print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Total text: {stats['total_text_length']:,} characters")
        print(f"Average filing length: {stats['average_filing_length']:,.0f} characters")
        print(f"Total sections: {stats['total_sections']:,}")
        
        print(f"\nFilings by company:")
        for ticker, count in list(stats['filings_by_company'].items())[:10]:
            print(f"  {ticker}: {count}")
            
        print(f"\nSections by type:")
        for section, count in list(stats['sections_by_type'].items())[:10]:
            print(f"  {section}: {count}")
            
        print(f"\nDatabase saved to: {db_path}")
        print(f"Parquet files in: {parquet_dir}/")
        print("="*60)
        
        # Example search
        print("\nExample search for 'artificial intelligence':")
        results = kb.search_content("artificial intelligence", limit=3)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['ticker']} ({result['fiscal_year']}) - {result['section_name']}")
            content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"   {content_preview}")
            
    finally:
        kb.close()


if __name__ == "__main__":
    main()

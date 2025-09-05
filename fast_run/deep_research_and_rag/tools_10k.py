from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
from smolagents import Tool


class TenKSearchTool(Tool):
    """
    Search tool for 10-K filings in DuckDB database.
    Returns structured search results from company filings.
    """
    
    name = "tenk_search"
    description = "Search through 10-K SEC filings by company, year, section, or content keywords."
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query - can be company names, business concepts, or specific topics.",
        },
        "tickers": {
            "type": "array",
            "description": "Optional list of company tickers to filter by (e.g., ['AAPL', 'MSFT']).",
            "items": {"type": "string"},
            "nullable": True,
            "default": None,
        },
        "years": {
            "type": "array", 
            "description": "Optional list of fiscal years to filter by (e.g., [2023, 2022]).",
            "items": {"type": "integer"},
            "nullable": True,
            "default": None,
        },
        "sections": {
            "type": "array",
            "description": "Optional list of section names to filter by (e.g., ['Business', 'Risk Factors']).",
            "items": {"type": "string"},
            "nullable": True,
            "default": None,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (1-20).",
            "nullable": True,
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
    }
    output_type = "object"

    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        super().__init__()
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            # Try relative to parent directory
            parent_db = Path(__file__).parent.parent / db_path
            if parent_db.exists():
                self.db_path = parent_db
            else:
                raise FileNotFoundError(f"10-K database not found at {db_path} or {parent_db}")

    def forward(
        self,
        query: str,
        tickers: Optional[List[str]] = None,
        years: Optional[List[int]] = None, 
        sections: Optional[List[str]] = None,
        limit: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """Search 10-K filings with optional filters."""
        
        if not query or not query.strip():
            return {"results": [], "message": "Query cannot be empty"}
            
        limit = max(1, min(limit or 5, 20))
        
        conn = duckdb.connect(str(self.db_path))
        
        try:
            # Build dynamic SQL query
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.filing_date,
                    f.fiscal_year,
                    s.section_name,
                    LENGTH(s.content) as content_length,
                    -- Create a preview around the search term
                    CASE 
                        WHEN position(lower(?) in lower(s.content)) > 0 THEN
                            substr(s.content, 
                                greatest(1, position(lower(?) in lower(s.content)) - 150),
                                400)
                        ELSE
                            substr(s.content, 1, 400)
                    END as snippet,
                    s.content
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE s.content ILIKE ?
            """
            
            params = [query, query, f'%{query}%']
            
            # Add optional filters
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
                
            sql += " ORDER BY f.filing_date DESC, f.ticker LIMIT ?"
            params.append(limit)
            
            # Execute query
            results = conn.execute(sql, params).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            # Format results
            formatted_results = []
            for row in results:
                result_dict = dict(zip(columns, row))
                
                # Clean up snippet
                snippet = result_dict.get('snippet', '')
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                result_dict['snippet'] = snippet
                
                # Remove full content from output (too large)
                result_dict.pop('content', None)
                
                # Create citation format
                result_dict['citation'] = f"TenK://{result_dict['ticker']}/{result_dict['fiscal_year']}/{result_dict['section_name']}"
                
                formatted_results.append(result_dict)
            
            return {
                "results": formatted_results,
                "query": query,
                "total_found": len(formatted_results),
                "filters_applied": {
                    "tickers": tickers,
                    "years": years, 
                    "sections": sections,
                },
            }
            
        except Exception as e:
            return {
                "results": [],
                "error": f"Database query failed: {str(e)}",
                "query": query,
            }
        finally:
            conn.close()


class TenKReadTool(Tool):
    """
    Read detailed content from a specific 10-K filing section.
    """
    
    name = "tenk_read"
    description = "Read the full content of a specific section from a 10-K filing."
    inputs = {
        "ticker": {
            "type": "string",
            "description": "Company ticker symbol (e.g., 'AAPL').",
        },
        "year": {
            "type": "integer",
            "description": "Fiscal year of the filing.",
        },
        "section": {
            "type": "string", 
            "description": "Section name to read (e.g., 'Business', 'Risk Factors').",
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return (default: 8000).",
            "nullable": True,
            "default": 8000,
            "minimum": 500,
            "maximum": 20000,
        },
    }
    output_type = "object"

    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        super().__init__()
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            # Try relative to parent directory
            parent_db = Path(__file__).parent.parent / db_path
            if parent_db.exists():
                self.db_path = parent_db
            else:
                raise FileNotFoundError(f"10-K database not found at {db_path} or {parent_db}")

    def forward(
        self,
        ticker: str,
        year: int,
        section: str,
        max_chars: Optional[int] = 8000,
    ) -> Dict[str, Any]:
        """Read full content of a specific 10-K section."""
        
        max_chars = max(500, min(max_chars or 8000, 20000))
        
        conn = duckdb.connect(str(self.db_path))
        
        try:
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.filing_date,
                    f.fiscal_year,
                    s.section_name,
                    s.content,
                    s.content_length,
                    s.word_count
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE f.ticker = ? 
                  AND f.fiscal_year = ?
                  AND s.section_name = ?
                LIMIT 1
            """
            
            result = conn.execute(sql, [ticker.upper(), year, section]).fetchone()
            
            if not result:
                return {
                    "found": False,
                    "message": f"No section '{section}' found for {ticker} in fiscal year {year}",
                    "ticker": ticker,
                    "year": year,
                    "section": section,
                }
            
            # Unpack result
            (ret_ticker, company_name, filing_date, fiscal_year, 
             section_name, content, content_length, word_count) = result
            
            # Truncate content if needed
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated...]"
            
            return {
                "found": True,
                "ticker": ret_ticker,
                "company_name": company_name,
                "filing_date": str(filing_date),
                "fiscal_year": fiscal_year,
                "section_name": section_name,
                "content": content,
                "content_length": content_length,
                "word_count": word_count,
                "citation": f"TenK://{ret_ticker}/{fiscal_year}/{section_name}",
                "truncated": len(content) >= max_chars,
            }
            
        except Exception as e:
            return {
                "found": False,
                "error": f"Database read failed: {str(e)}",
                "ticker": ticker,
                "year": year,
                "section": section,
            }
        finally:
            conn.close()


__all__ = ["TenKSearchTool", "TenKReadTool"]
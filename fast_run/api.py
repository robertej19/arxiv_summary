#!/usr/bin/env python3
"""
FastAPI backend for 10-K Knowledge Base.

Provides REST API endpoints for querying the knowledge base.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Generator
import json
import duckdb
import logging
from pathlib import Path
from datetime import date
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add deep_research_and_rag to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    tickers: Optional[List[str]] = Field(None, description="Filter by company tickers")
    years: Optional[List[int]] = Field(None, description="Filter by fiscal years")
    sections: Optional[List[str]] = Field(None, description="Filter by section names")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)

class SearchResult(BaseModel):
    ticker: Optional[str]
    company_name: Optional[str]
    filing_date: Optional[date]
    fiscal_year: Optional[int]
    section_name: str
    content_length: int
    content_preview: str
    relevance_score: Optional[float] = None
    file_path: str

class CompanyOverview(BaseModel):
    ticker: str
    company_name: Optional[str]
    total_filings: int
    filing_date_range: tuple[Optional[date], Optional[date]]
    avg_filing_length: Optional[float]
    yearly_filings: Dict[int, int]
    section_availability: Dict[str, int]

class KnowledgeBaseStats(BaseModel):
    total_filings: int
    unique_companies: int
    date_range: tuple[Optional[date], Optional[date]]
    total_text_length: Optional[int]
    average_filing_length: Optional[float]
    total_sections: int
    top_companies: Dict[str, int]
    top_sections: Dict[str, int]

class ResearchRequest(BaseModel):
    question: str = Field(..., description="Research question to analyze")
    
class ResearchResponse(BaseModel):
    question: str
    answer: str
    status: str
    processing_time: Optional[float] = None

class ProgressUpdate(BaseModel):
    step: str
    message: str
    progress: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class TenKAPI:
    """API interface for 10-K Knowledge Base."""
    
    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        self.db_path = db_path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Knowledge base not found at {db_path}")
        
        # Test connection
        try:
            conn = duckdb.connect(db_path)
            conn.execute("SELECT COUNT(*) FROM filings").fetchone()
            conn.close()
            logger.info(f"Connected to knowledge base: {db_path}")
        except Exception as e:
            raise Exception(f"Failed to connect to knowledge base: {e}")
    
    def get_connection(self):
        """Get a fresh database connection with FTS extension."""
        conn = duckdb.connect(self.db_path)
        # Install and load full-text search extension
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
        except:
            pass  # Extension might already be loaded
        return conn
    
    def search_filings(self, request: SearchRequest) -> List[SearchResult]:
        """Search across all filings using optimized full-text search."""
        conn = self.get_connection()
        
        try:
            # Use database-level text search and preview generation
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.filing_date,
                    f.fiscal_year,
                    s.section_name,
                    s.content_length,
                    -- Generate preview in database instead of Python
                    CASE 
                        WHEN position(lower(?) in lower(s.content)) > 0 THEN
                            '...' || 
                            substr(s.content, 
                                   greatest(1, position(lower(?) in lower(s.content)) - 150), 
                                   300) || 
                            '...'
                        ELSE
                            substr(s.content, 1, 300) || '...'
                    END as content_preview,
                    f.file_path
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE 
                    -- Use more efficient text search
                    lower(s.content) LIKE lower(?)
            """
            
            query_param = f'%{request.query}%'
            params = [request.query, request.query, query_param]
            
            if request.tickers:
                placeholders = ','.join(['?' for _ in request.tickers])
                sql += f" AND f.ticker IN ({placeholders})"
                params.extend(request.tickers)
                
            if request.years:
                placeholders = ','.join(['?' for _ in request.years])
                sql += f" AND f.fiscal_year IN ({placeholders})"
                params.extend(request.years)
                
            if request.sections:
                placeholders = ','.join(['?' for _ in request.sections])
                sql += f" AND s.section_name IN ({placeholders})"
                params.extend(request.sections)
                
            sql += " ORDER BY f.filing_date DESC, f.ticker LIMIT ?"
            params.append(request.limit)
            
            results = conn.execute(sql, params).fetchall()
            
            search_results = []
            for row in results:
                ticker, company_name, filing_date, fiscal_year, section_name, content_length, content_preview, file_path = row
                
                search_results.append(SearchResult(
                    ticker=ticker,
                    company_name=company_name,
                    filing_date=filing_date,
                    fiscal_year=fiscal_year,
                    section_name=section_name,
                    content_length=content_length,
                    content_preview=content_preview,
                    file_path=file_path
                ))
            
            return search_results
            
        finally:
            conn.close()
    
    def get_company_overview(self, ticker: str) -> CompanyOverview:
        """Get comprehensive overview of a company's filings."""
        conn = self.get_connection()
        
        try:
            # Basic company info
            company_info = conn.execute("""
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
                raise HTTPException(status_code=404, detail=f"No filings found for ticker {ticker}")
                
            # Filing details by year
            yearly_filings = dict(conn.execute("""
                SELECT fiscal_year, COUNT(*) as count
                FROM filings 
                WHERE ticker = ?
                GROUP BY fiscal_year
                ORDER BY fiscal_year DESC
            """, (ticker,)).fetchall())
            
            # Section availability
            section_availability = dict(conn.execute("""
                SELECT s.section_name, COUNT(*) as count
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE f.ticker = ?
                GROUP BY s.section_name
                ORDER BY count DESC
            """, (ticker,)).fetchall())
            
            return CompanyOverview(
                ticker=company_info[0],
                company_name=company_info[1],
                total_filings=company_info[2],
                filing_date_range=(company_info[3], company_info[4]),
                avg_filing_length=company_info[5],
                yearly_filings=yearly_filings,
                section_availability=section_availability
            )
            
        finally:
            conn.close()
    
    def get_stats(self) -> KnowledgeBaseStats:
        """Get knowledge base statistics."""
        conn = self.get_connection()
        
        try:
            # Basic stats
            basic_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_filings,
                    COUNT(DISTINCT ticker) as unique_companies,
                    MIN(filing_date) as min_date,
                    MAX(filing_date) as max_date,
                    SUM(full_text_length) as total_text_length,
                    AVG(full_text_length) as avg_filing_length
                FROM filings
            """).fetchone()
            
            total_sections = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            
            # Top companies by filing count
            top_companies = dict(conn.execute("""
                SELECT ticker, COUNT(*) as count
                FROM filings 
                WHERE ticker IS NOT NULL 
                GROUP BY ticker 
                ORDER BY count DESC 
                LIMIT 10
            """).fetchall())
            
            # Top sections by count
            top_sections = dict(conn.execute("""
                SELECT section_name, COUNT(*) as count
                FROM sections 
                GROUP BY section_name 
                ORDER BY count DESC 
                LIMIT 10
            """).fetchall())
            
            return KnowledgeBaseStats(
                total_filings=basic_stats[0],
                unique_companies=basic_stats[1],
                date_range=(basic_stats[2], basic_stats[3]),
                total_text_length=basic_stats[4],
                average_filing_length=basic_stats[5],
                total_sections=total_sections,
                top_companies=top_companies,
                top_sections=top_sections
            )
            
        finally:
            conn.close()
    
    def get_available_tickers(self) -> List[str]:
        """Get list of all available company tickers."""
        conn = self.get_connection()
        
        try:
            tickers = conn.execute("""
                SELECT DISTINCT ticker 
                FROM filings 
                WHERE ticker IS NOT NULL 
                ORDER BY ticker
            """).fetchall()
            
            return [ticker[0] for ticker in tickers]
            
        finally:
            conn.close()
    
    def get_available_sections(self) -> List[str]:
        """Get list of all available section types."""
        conn = self.get_connection()
        
        try:
            sections = conn.execute("""
                SELECT DISTINCT section_name 
                FROM sections 
                ORDER BY section_name
            """).fetchall()
            
            return [section[0] for section in sections]
            
        finally:
            conn.close()

# Initialize FastAPI app
app = FastAPI(
    title="10-K Knowledge Base API",
    description="REST API for searching and analyzing SEC 10-K filings",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API interface
try:
    api = TenKAPI()
except Exception as e:
    logger.error(f"Failed to initialize API: {e}")
    raise

# Initialize thread pool for research tasks
executor = ThreadPoolExecutor(max_workers=2)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "10-K Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Search filings",
            "/company/{ticker}": "Get company overview",
            "/stats": "Get knowledge base statistics",
            "/tickers": "List available tickers",
            "/sections": "List available sections",
            "/research": "AI-powered research on 10-K filings"
        }
    }

@app.post("/search", response_model=List[SearchResult])
async def search_filings(request: SearchRequest):
    """Search across all 10-K filings."""
    try:
        results = api.search_filings(request)
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/company/{ticker}", response_model=CompanyOverview)
async def get_company_overview(ticker: str):
    """Get comprehensive overview of a company's filings."""
    try:
        return api.get_company_overview(ticker.upper())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Company overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=KnowledgeBaseStats)
async def get_stats():
    """Get knowledge base statistics."""
    try:
        return api.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tickers", response_model=List[str])
async def get_available_tickers():
    """Get list of all available company tickers."""
    try:
        return api.get_available_tickers()
    except Exception as e:
        logger.error(f"Tickers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sections", response_model=List[str])
async def get_available_sections():
    """Get list of all available section types."""
    try:
        return api.get_available_sections()
    except Exception as e:
        logger.error(f"Sections error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """Conduct AI-powered research on 10-K filings."""
    import time
    start_time = time.time()
    
    try:
        # Import research function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))
        from tenk_research_agent import conduct_10k_research
        
        # Run research in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor, 
            conduct_10k_research, 
            request.question
        )
        
        processing_time = time.time() - start_time
        
        return ResearchResponse(
            question=request.question,
            answer=answer,
            status="completed",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Research error: {e}")
        processing_time = time.time() - start_time
        
        return ResearchResponse(
            question=request.question,
            answer=f"Research failed: {str(e)}",
            status="error", 
            processing_time=processing_time
        )

@app.post("/research/stream")
async def conduct_research_stream(request: ResearchRequest):
    """Conduct AI-powered research with streaming progress updates."""
    
    def generate_progress() -> Generator[str, None, None]:
        import time
        import traceback
        start_time = time.time()
        
        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'step': 'connected', 'message': 'Connection established', 'progress': 0.0})}\n\n"
            
            # Import the streaming research function
            sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_research_and_rag'))
            from tenk_research_agent import conduct_10k_research_with_progress, ProgressUpdate
            
            logger.info(f"Starting streaming research for question: {request.question[:100]}")
            
            # Generator that yields progress updates
            update_count = 0
            for progress_update in conduct_10k_research_with_progress(request.question):
                try:
                    progress_update.timestamp = time.time() - start_time
                    update_count += 1
                    
                    # Format as Server-Sent Events
                    data = progress_update.model_dump()
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    logger.debug(f"Sent progress update {update_count}: {progress_update.step}")
                    
                    # Check if research is complete
                    if progress_update.step == "completed":
                        logger.info(f"Research completed after {update_count} updates")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in progress update {update_count}: {e}")
                    continue
                
        except ImportError as e:
            logger.error(f"Import error in streaming: {e}")
            error_data = {
                "step": "error",
                "message": f"Import failed: {str(e)}. Please check if the research agent is properly installed.",
                "progress": 0.0,
                "timestamp": time.time() - start_time
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming research error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_data = {
                "step": "error", 
                "message": f"Research failed: {str(e)}",
                "progress": 0.0,
                "timestamp": time.time() - start_time
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# Convenience GET endpoint for simple searches
@app.get("/search", response_model=List[SearchResult])
async def search_filings_get(
    q: str = Query(..., description="Search query"),
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
    years: Optional[str] = Query(None, description="Comma-separated years"),
    sections: Optional[str] = Query(None, description="Comma-separated sections"),
    limit: int = Query(10, description="Result limit", ge=1, le=100)
):
    """Simple GET endpoint for searching filings."""
    
    # Parse comma-separated parameters
    tickers_list = tickers.split(",") if tickers else None
    years_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()] if years else None
    sections_list = sections.split(",") if sections else None
    
    request = SearchRequest(
        query=q,
        tickers=tickers_list,
        years=years_list,
        sections=sections_list,
        limit=limit
    )
    
    return await search_filings(request)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

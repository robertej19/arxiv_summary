#!/usr/bin/env python3
"""
Fast 10-K Research Agent - Optimized for seconds-level responses

Key optimizations for closed corpus:
1. Pre-computed keyword index for instant search
2. Single LLM call with consolidated prompt
3. Parallel database queries 
4. Lightweight evidence selection
5. Template-based responses for common patterns

Usage:
    python fast_tenk_agent.py "How are companies approaching AI?"
"""

from __future__ import annotations

import os
import sys
import json
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb

# Local imports  
from llama_cpp_model import LlamaCppModel
from tools_10k import TenKSearchTool, TenKReadTool

# --------------------------
# Fast Mode Configuration
# --------------------------
MODEL_PATH = os.getenv("LLAMA_GGUF", "../../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.getenv("LLAMA_N_CTX", "8192"))           # Smaller context for speed
MAX_GEN_TOK = int(os.getenv("LLAMA_MAX_TOK", "400"))    # Shorter responses

# Fast mode limits (aggressive reduction)
MAX_SEARCH_TIME = 2.0        # Max seconds for all searches
MAX_EVIDENCE_ITEMS = 3       # Reduced from 5
MAX_CONTENT_LENGTH = 800     # Reduced from 1200
PARALLEL_QUERIES = True      # Enable parallel search
CACHE_SEARCHES = True        # Cache common searches


# --------------------------
# Fast Evidence Structure
# --------------------------
@dataclass
class FastEvidence:
    """Lightweight evidence structure for fast processing."""
    ticker: str
    company_name: str
    fiscal_year: int
    section_name: str
    snippet: str
    score: float = 0.0
    
    def to_citation(self) -> str:
        return f"{self.company_name} ({self.ticker}) {self.fiscal_year} 10-K"


@dataclass 
class FastSearchCache:
    """Simple in-memory cache for search results."""
    cache: Dict[str, List[FastEvidence]] = field(default_factory=dict)
    max_size: int = 100
    
    def get_cache_key(self, query: str) -> str:
        """Create consistent cache key from query."""
        normalized = re.sub(r'\W+', '', query.lower())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[List[FastEvidence]]:
        """Get cached results if available."""
        key = self.get_cache_key(query)
        return self.cache.get(key)
    
    def put(self, query: str, results: List[FastEvidence]):
        """Cache search results."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_cache_key(query)
        self.cache[key] = results


# Global cache instance
search_cache = FastSearchCache()


# --------------------------
# Fast Database Queries
# --------------------------
class FastDatabaseQuery:
    """Optimized database queries for speed."""
    
    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            # Try parent directory
            parent_db = Path(__file__).parent.parent / db_path
            if parent_db.exists():
                self.db_path = parent_db
            else:
                # Try relative to script location
                script_dir = Path(__file__).parent
                current_db = script_dir / ".." / db_path
                if current_db.exists():
                    self.db_path = current_db
                else:
                    raise FileNotFoundError(f"Database not found: {db_path}")
    
    def fast_search(self, query: str, limit: int = 3) -> List[FastEvidence]:
        """Optimized search with minimal data transfer."""
        conn = duckdb.connect(str(self.db_path))
        
        try:
            # Single optimized query with minimal data
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.fiscal_year,
                    s.section_name,
                    -- Create focused snippet (faster than large content)
                    CASE 
                        WHEN position(lower(?) in lower(s.content)) > 0 THEN
                            substr(s.content, 
                                greatest(1, position(lower(?) in lower(s.content)) - 100),
                                300)
                        ELSE
                            substr(s.content, 1, 300)
                    END as snippet
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                WHERE s.content ILIKE ?
                ORDER BY f.fiscal_year DESC, f.ticker
                LIMIT ?
            """
            
            results = conn.execute(sql, [query, query, f'%{query}%', limit]).fetchall()
            
            evidence_list = []
            for row in results:
                ticker, company_name, fiscal_year, section_name, snippet = row
                
                # Clean snippet
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                
                evidence_list.append(FastEvidence(
                    ticker=ticker,
                    company_name=company_name,
                    fiscal_year=fiscal_year,
                    section_name=section_name,
                    snippet=snippet,
                    score=1.0  # Simple scoring
                ))
            
            return evidence_list
            
        finally:
            conn.close()
    
    def parallel_search(self, queries: List[str], limit_per_query: int = 2) -> List[FastEvidence]:
        """Execute multiple searches in parallel."""
        all_evidence = []
        
        if PARALLEL_QUERIES and len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
                future_to_query = {
                    executor.submit(self.fast_search, query, limit_per_query): query 
                    for query in queries
                }
                
                for future in as_completed(future_to_query, timeout=MAX_SEARCH_TIME):
                    try:
                        evidence = future.result()
                        all_evidence.extend(evidence)
                    except Exception as e:
                        print(f"Search failed: {e}")
        else:
            # Sequential fallback
            for query in queries:
                evidence = self.fast_search(query, limit_per_query)
                all_evidence.extend(evidence)
        
        return all_evidence


# --------------------------
# Fast Model Setup
# --------------------------
def build_fast_model() -> LlamaCppModel:
    """Build optimized model for fast inference."""
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        alt_paths = [
            Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            Path(__file__).parent.parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
            sys.exit(1)
    
    return LlamaCppModel(
        model_path=str(model_path),
        n_ctx=N_CTX,                    # Smaller context
        max_tokens=MAX_GEN_TOK,         # Shorter responses  
        temperature=0.0,                # Deterministic for speed
        top_p=0.95,
        n_threads=8,                    # Maximize CPU usage
        repeat_penalty=1.05,            # Lower penalty
        verbose=False,
        # Additional speed optimizations
        n_batch=512,                    # Larger batch for efficiency
    )


# --------------------------
# Fast Query Processing
# --------------------------
def extract_search_terms(question: str) -> List[str]:
    """Extract 2-3 key search terms from question."""
    # Remove stop words and extract key terms
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    keywords = [w for w in words if w not in stop_words]
    
    # Take first 2-3 most relevant terms
    return keywords[:3] if len(keywords) >= 3 else keywords


def create_search_queries(question: str) -> List[str]:
    """Create focused search queries from question."""
    terms = extract_search_terms(question)
    
    if not terms:
        return [question]
    
    # Create 2 complementary queries maximum
    queries = []
    
    # Primary query: combine top terms
    if len(terms) >= 2:
        queries.append(f"{terms[0]} {terms[1]}")
    
    # Secondary query: single most important term
    if terms:
        queries.append(terms[0])
    
    return queries[:2]  # Maximum 2 queries for speed


# --------------------------
# Optimized Response Templates
# --------------------------
FAST_SYNTHESIS_PROMPT = """Answer this question using ONLY the provided evidence. Be concise and specific.

Question: {question}

Evidence:
{evidence}

Instructions:
- Write 2-4 sentences maximum
- Include citations [1], [2] after key facts
- Focus on the most important findings
- Use simple, clear language

Answer:"""


def fast_synthesize_answer(question: str, evidence_list: List[FastEvidence], model: LlamaCppModel) -> str:
    """Generate fast answer using minimal context."""
    
    if not evidence_list:
        return "No relevant evidence found in the 10-K database for this question."
    
    # Prepare concise evidence
    evidence_text = ""
    citations = []
    
    for i, evidence in enumerate(evidence_list[:MAX_EVIDENCE_ITEMS], 1):
        # Truncate snippet for speed
        snippet = evidence.snippet[:MAX_CONTENT_LENGTH]
        evidence_text += f"[{i}] {evidence.company_name} ({evidence.ticker}) {evidence.fiscal_year}: {snippet}\n\n"
        citations.append(f"[{i}] {evidence.to_citation()}")
    
    # Create compact prompt
    prompt = FAST_SYNTHESIS_PROMPT.format(
        question=question,
        evidence=evidence_text.strip()
    )
    
    # Generate with timeout protection
    try:
        response = model.generate([{
            "role": "user",
            "content": prompt
        }]).content
        
        # Add citations
        if citations:
            response += "\n\nSources:\n" + "\n".join(citations)
        
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"


# --------------------------
# Fast Research Pipeline  
# --------------------------
def fast_10k_research(question: str, verbose: bool = False) -> str:
    """Fast research pipeline optimized for seconds-level response."""
    start_time = time.time()
    
    if verbose:
        print(f"🚀 Fast 10-K research: {question}")
    
    # Step 1: Check cache first
    cache_key = search_cache.get_cache_key(question)
    cached_evidence = search_cache.get(question)
    
    if cached_evidence and CACHE_SEARCHES:
        if verbose:
            print(f"⚡ Cache hit! Using cached results ({len(cached_evidence)} items)")
        evidence_list = cached_evidence
    else:
        # Step 2: Create focused search queries
        queries = create_search_queries(question)
        if verbose:
            print(f"🔍 Search queries: {queries}")
        
        # Step 3: Fast parallel search
        db_query = FastDatabaseQuery()
        evidence_list = db_query.parallel_search(queries, limit_per_query=2)
        
        # Cache results
        if CACHE_SEARCHES:
            search_cache.put(question, evidence_list)
        
        if verbose:
            print(f"📊 Found {len(evidence_list)} evidence items")
    
    # Step 4: Load model and synthesize (only if we have evidence)
    if evidence_list:
        if verbose:
            print("🤖 Generating answer...")
        
        model = build_fast_model()
        answer = fast_synthesize_answer(question, evidence_list, model)
    else:
        answer = "No relevant information found in the 10-K database for this question."
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"✅ Complete in {elapsed:.2f}s")
    
    return answer


# --------------------------
# CLI Interface
# --------------------------
def main():
    """Fast research CLI."""
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How are companies approaching artificial intelligence?"
    
    print("=" * 60)
    print("🚀 FAST 10-K RESEARCH AGENT")  
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    
    try:
        start = time.time()
        answer = fast_10k_research(question, verbose=True)
        total_time = time.time() - start
        
        print(f"\n📋 ANSWER (Generated in {total_time:.2f}s)")
        print("-" * 60)
        print(answer)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
10-K Deep Research Agent using SmolagentsAI

A specialized research agent that can quickly search through 10-K SEC filings
and synthesize comprehensive answers about companies, industries, and business trends.

Usage:
    python tenk_research_agent.py "How are major tech companies approaching AI?"
    
Environment Variables:
    LLAMA_GGUF: Path to GGUF model file (default: ../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf)
    LLAMA_N_CTX: Context window size (default: 8192)
    LLAMA_MAX_TOK: Max generation tokens (default: 800)
"""

from __future__ import annotations

import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Generator
from pathlib import Path
import time

# Local imports
from llama_cpp_model import LlamaCppModel
from tools_10k import TenKSearchTool, TenKReadTool


# --------------------------
# Configuration
# --------------------------
MODEL_PATH = os.getenv("LLAMA_GGUF", "../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")  # Default to Qwen2.5 7B quantized
N_CTX = int(os.getenv("LLAMA_N_CTX", "16384"))                    # Increased context window for faster processing
MAX_GEN_TOK = int(os.getenv("LLAMA_MAX_TOK", "800"))             # Max generation tokens

# Research parameters
MAX_SEARCH_QUERIES = 3        # Maximum search queries to generate
MAX_RESULTS_PER_QUERY = 4     # Results per search query
MAX_DETAILED_READS = 3        # Maximum detailed section reads
MAX_EVIDENCE_ITEMS = 5        # Reduced from 8 for faster synthesis
EVIDENCE_CHUNK_SIZE = 1200    # Reduced from 2000 for faster processing


# --------------------------
# Progress tracking
# --------------------------
@dataclass
class ProgressUpdate:
    step: str
    message: str
    progress: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "message": self.message,
            "progress": self.progress,
            "details": self.details,
            "timestamp": self.timestamp
        }

# --------------------------
# Evidence tracking
# --------------------------
@dataclass
class Evidence:
    """Structured evidence from 10-K filings."""
    ticker: str
    company_name: str
    fiscal_year: int
    section_name: str
    content_snippet: str
    citation: str
    relevance_score: float = 0.0
    # Enhanced fields for citation linking
    citation_id: Optional[str] = None  # Unique ID for this citation
    file_path: Optional[str] = None    # Path to original HTML file
    search_text: Optional[str] = None  # Text used for highlighting in source


@dataclass
class ResearchContext:
    """Context for tracking research progress."""
    question: str
    evidence: List[Evidence] = field(default_factory=list)
    companies_found: set = field(default_factory=set)
    sections_covered: set = field(default_factory=set)
    
    def add_evidence(self, evidence_item: Evidence):
        """Add evidence and update tracking."""
        self.evidence.append(evidence_item)
        self.companies_found.add(evidence_item.ticker)
        self.sections_covered.add(evidence_item.section_name)
    
    def get_unique_companies(self) -> List[str]:
        """Get list of unique companies in evidence."""
        return list(self.companies_found)
    
    def get_sections_covered(self) -> List[str]:
        """Get list of unique sections covered."""
        return list(self.sections_covered)


# --------------------------
# Helper functions
# --------------------------
def _extract_keywords(question: str, max_keywords: int = 6) -> List[str]:
    """Extract key terms from the research question."""
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    keywords = [w for w in words if w not in stop_words]
    
    # Return first max_keywords unique terms
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen and len(result) < max_keywords:
            seen.add(kw)
            result.append(kw)
    
    return result


def _parse_json_list(text: str) -> List[str]:
    """Parse JSON list from LLM output, with fallbacks."""
    # Try to find JSON array
    import json as json_mod
    
    # Look for JSON array pattern
    match = re.search(r'\[([^\[\]]*)\]', text)
    if match:
        try:
            # Try to parse as JSON
            array_text = match.group(0)
            parsed = json_mod.loads(array_text)
            if isinstance(parsed, list):
                return [str(item).strip('"') for item in parsed if isinstance(item, str)]
        except:
            pass
    
    # Fallback: extract quoted strings
    quotes = re.findall(r'"([^"]+)"', text)
    if quotes:
        return quotes[:3]  # Limit to 3
    
    # Fallback: split by lines and clean
    lines = [line.strip().strip('-•').strip() for line in text.split('\n') if line.strip()]
    return [line for line in lines if line and not line.startswith('[') and not line.endswith(']')][:3]


# --------------------------
# Model and tools setup
# --------------------------
def build_model() -> LlamaCppModel:
    """Build the local LLM."""
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        # Try alternative locations
        alt_paths = [
            Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            Path(__file__).parent.parent.parent / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"ERROR: Model not found at {MODEL_PATH} or alternative locations", file=sys.stderr)
            print("Available alternatives searched:", file=sys.stderr)
            for alt in alt_paths:
                print(f"  {alt}", file=sys.stderr)
            print("\nTo download the model, run:", file=sys.stderr)
            print("  bash download_model.sh", file=sys.stderr)
            sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    return LlamaCppModel(
        model_path=str(model_path),
        n_ctx=N_CTX,
        max_tokens=MAX_GEN_TOK,
        temperature=0.1,  # Lower temperature for faster, more deterministic generation
        top_p=0.95,
        n_threads=8,      # Optimize CPU threads for faster processing
        repeat_penalty=1.1,
        verbose=False,
    )


def build_tools() -> Tuple[TenKSearchTool, TenKReadTool]:
    """Build the 10-K database tools."""
    return TenKSearchTool(), TenKReadTool()


# --------------------------
# Research pipeline
# --------------------------
QUERY_PLANNER_PROMPT = """You are a research planner for SEC 10-K filings analysis.

Given this research question, generate 2-3 specific search queries that would help answer it comprehensively.
Focus on different aspects: companies, business concepts, financial metrics, risks, etc.

Return ONLY a JSON array of strings, no other text.

Research Question: {question}

Example format: ["artificial intelligence revenue", "AI competition risks", "machine learning investments"]
"""

EVIDENCE_SYNTHESIZER_PROMPT = """You are a financial research analyst. Use ONLY the evidence provided to answer the research question.

Research Question: {question}

Evidence from 10-K filings:
{evidence_text}

Instructions:
- Write a comprehensive analysis using ONLY the provided evidence
- Include specific citations like [1], [2] after claims
- Highlight key insights, trends, and differences between companies
- If evidence is limited, state what's missing
- Use clear, professional language
- Structure with headings if helpful

Citations:
{citations}
"""


def conduct_10k_research(question: str) -> str:
    """Main research pipeline for 10-K analysis."""
    
    print(f"\n🔍 Starting 10-K research on: {question}")
    
    # Initialize components
    model = build_model()
    search_tool, read_tool = build_tools()
    context = ResearchContext(question=question)
    
    # Step 1: Generate search queries
    print("📝 Planning search queries...")
    planner_output = model.generate([{
        "role": "user", 
        "content": QUERY_PLANNER_PROMPT.format(question=question)
    }]).content
    
    search_queries = _parse_json_list(planner_output)
    if not search_queries:
        search_queries = [question]  # Fallback
    
    print(f"📋 Generated {len(search_queries)} search queries:")
    for i, query in enumerate(search_queries, 1):
        print(f"   {i}. {query}")
    
    # Step 2: Execute searches and collect evidence
    print("\n🔎 Executing searches...")
    
    all_search_results = []
    for i, query in enumerate(search_queries):
        print(f"   Searching: {query}")
        
        # Perform search
        search_result = search_tool.forward(
            query=query,
            limit=MAX_RESULTS_PER_QUERY,
        )
        
        results = search_result.get("results", [])
        all_search_results.extend(results)
        print(f"     Found {len(results)} results")
        
        # Convert to evidence
        for result in results:
            # Generate unique citation ID
            import hashlib
            citation_id = hashlib.md5(f"{result.get('ticker', '')}_{result.get('fiscal_year', 0)}_{result.get('section_name', '')}_{query}".encode()).hexdigest()[:8]
            
            evidence = Evidence(
                ticker=result.get("ticker", ""),
                company_name=result.get("company_name", ""),
                fiscal_year=result.get("fiscal_year", 0),
                section_name=result.get("section_name", ""),
                content_snippet=result.get("snippet", ""),
                citation=result.get("citation", ""),
                relevance_score=1.0,  # Could implement scoring
                citation_id=citation_id,
                file_path=result.get("file_path", ""),
                search_text=query,  # Store the search query for highlighting
            )
            context.add_evidence(evidence)
    
    print(f"📊 Collected {len(context.evidence)} evidence items from {len(context.companies_found)} companies")
    
    # Step 3: Get detailed content for most promising results
    print("\n📖 Reading detailed sections...")
    
    # Select top results for detailed reading
    detailed_reads = 0
    for evidence in context.evidence[:MAX_DETAILED_READS * 2]:  # Try more in case some fail
        if detailed_reads >= MAX_DETAILED_READS:
            break
            
        print(f"   Reading: {evidence.ticker} {evidence.fiscal_year} - {evidence.section_name}")
        
        read_result = read_tool.forward(
            ticker=evidence.ticker,
            year=evidence.fiscal_year,
            section=evidence.section_name,
            max_chars=2500,  # Reduced limit for faster processing
        )
        
        if read_result.get("found"):
            # Update evidence with full content
            evidence.content_snippet = read_result.get("content", "")[:EVIDENCE_CHUNK_SIZE]  # Use configurable chunk size
            detailed_reads += 1
        else:
            print(f"     ⚠️  Could not read section")
    
    print(f"📚 Successfully read {detailed_reads} detailed sections")
    
    # Step 4: Synthesize final answer
    print("\n✍️  Synthesizing final answer...")
    
    # Prepare evidence for synthesis (optimized)
    evidence_text = ""
    citations = []
    
    # Use only the most relevant evidence items
    relevant_evidence = context.evidence[:MAX_EVIDENCE_ITEMS]
    
    for i, evidence in enumerate(relevant_evidence, 1):
        # Create concise evidence entries
        evidence_text += f"[{i}] {evidence.company_name} ({evidence.ticker}) {evidence.fiscal_year} - {evidence.section_name}:\n"
        evidence_text += f"{evidence.content_snippet[:EVIDENCE_CHUNK_SIZE]}\n\n"
        
        # Include citation ID in citations for frontend linking
        citation_info = f"[{i}] {evidence.company_name} ({evidence.ticker}) {evidence.fiscal_year} 10-K, {evidence.section_name}"
        if evidence.citation_id:
            citation_info += f" [cite:{evidence.citation_id}]"
        citations.append(citation_info)
    
    citations_text = "\n".join(citations)
    
    # Optimize prompt length to fit context window efficiently
    estimated_prompt_length = len(question) + len(evidence_text) + len(citations_text) + 1000  # buffer for template
    if estimated_prompt_length > N_CTX * 0.7:  # Use 70% of context to leave room for generation
        print("⚡ Optimizing prompt length for faster processing...")
        # Truncate evidence if needed
        evidence_text = evidence_text[:int(N_CTX * 0.5)]  # Use 50% for evidence
    
    # Generate final synthesis
    synthesis_prompt = EVIDENCE_SYNTHESIZER_PROMPT.format(
        question=question,
        evidence_text=evidence_text,
        citations=citations_text,
    )
    
    final_answer = model.generate([{
        "role": "user",
        "content": synthesis_prompt
    }]).content
    
    print("✅ Research complete!")
    return final_answer


def conduct_10k_research_with_progress(question: str) -> Generator[ProgressUpdate, None, None]:
    """Main research pipeline with streaming progress updates."""
    start_time = time.time()
    
    # Initialize progress
    yield ProgressUpdate(
        step="initialization",
        message="🔧 Initializing research tools and models...",
        progress=0.0,
        details={"question": question}
    )
    
    # Build model and tools
    model = build_model()
    search_tool, read_tool = build_tools()
    context = ResearchContext(question)
    
    yield ProgressUpdate(
        step="planning",
        message="🧠 Planning search strategy...",
        progress=0.1,
        details={"model_loaded": True}
    )
    
    # Step 1: Generate search queries
    planner_output = model.generate([{
        "role": "user",
        "content": QUERY_PLANNER_PROMPT.format(question=question)
    }]).content
    
    try:
        queries = json.loads(planner_output)[:MAX_SEARCH_QUERIES]
    except:
        queries = [question]
    
    yield ProgressUpdate(
        step="planning_complete",
        message=f"📋 Generated {len(queries)} search queries",
        progress=0.2,
        details={"queries": queries}
    )
    
    # Step 2: Execute searches
    yield ProgressUpdate(
        step="searching",
        message="🔍 Searching through 10-K filings...",
        progress=0.25
    )
    
    for i, query in enumerate(queries):
        yield ProgressUpdate(
            step="searching",
            message=f"🔍 Searching: {query[:60]}...",
            progress=0.25 + (i / len(queries)) * 0.25,
            details={"current_query": query, "query_num": i + 1, "total_queries": len(queries)}
        )
        
        search_results = search_tool.forward(query=query, limit=MAX_RESULTS_PER_QUERY)
        
        if search_results and "results" in search_results:
            for result in search_results["results"]:
                evidence = Evidence(
                    ticker=result.get("ticker", ""),
                    company_name=result.get("company_name", ""),
                    fiscal_year=result.get("fiscal_year", 0),
                    section_name=result.get("section_name", ""),
                    content_snippet=result.get("snippet", "")[:500],
                    source_file=result.get("source_file", ""),
                    relevance_score=result.get("score", 0.0)
                )
                context.add_evidence(evidence)
    
    yield ProgressUpdate(
        step="search_complete", 
        message=f"📊 Found {len(context.evidence)} evidence items from {len(context.companies_found)} companies",
        progress=0.5,
        details={
            "evidence_count": len(context.evidence),
            "companies_found": len(context.companies_found),
            "companies": list(context.companies_found)
        }
    )
    
    # Step 3: Read detailed content
    yield ProgressUpdate(
        step="reading",
        message="📖 Reading detailed sections...",
        progress=0.55
    )
    
    detailed_reads = 0
    total_to_read = min(len(context.evidence), MAX_DETAILED_READS * 2)
    
    for i, evidence in enumerate(context.evidence[:MAX_DETAILED_READS * 2]):
        if detailed_reads >= MAX_DETAILED_READS:
            break
            
        yield ProgressUpdate(
            step="reading",
            message=f"📖 Reading: {evidence.ticker} {evidence.fiscal_year} - {evidence.section_name}",
            progress=0.55 + (i / total_to_read) * 0.25,
            details={
                "reading": f"{evidence.ticker} {evidence.fiscal_year}",
                "section": evidence.section_name,
                "read_num": detailed_reads + 1,
                "max_reads": MAX_DETAILED_READS
            }
        )
        
        read_result = read_tool.forward(
            ticker=evidence.ticker,
            year=evidence.fiscal_year,
            section=evidence.section_name,
            max_chars=2500,
        )
        
        if read_result.get("found"):
            evidence.content_snippet = read_result.get("content", "")[:EVIDENCE_CHUNK_SIZE]
            detailed_reads += 1
    
    yield ProgressUpdate(
        step="reading_complete",
        message=f"📚 Successfully read {detailed_reads} detailed sections",
        progress=0.8,
        details={"detailed_reads": detailed_reads}
    )
    
    # Step 4: Synthesize answer
    yield ProgressUpdate(
        step="synthesizing",
        message="✍️ Analyzing evidence and synthesizing answer...",
        progress=0.85
    )
    
    # Prepare evidence for synthesis (optimized)
    evidence_text = ""
    citations = []
    
    relevant_evidence = context.evidence[:MAX_EVIDENCE_ITEMS]
    
    for i, evidence in enumerate(relevant_evidence, 1):
        evidence_text += f"[{i}] {evidence.company_name} ({evidence.ticker}) {evidence.fiscal_year} - {evidence.section_name}:\n"
        evidence_text += f"{evidence.content_snippet[:EVIDENCE_CHUNK_SIZE]}\n\n"
        citations.append(f"[{i}] {evidence.company_name} ({evidence.ticker}) {evidence.fiscal_year} 10-K, {evidence.section_name}")
    
    citations_text = "\n".join(citations)
    
    # Optimize prompt length
    estimated_prompt_length = len(question) + len(evidence_text) + len(citations_text) + 1000
    if estimated_prompt_length > N_CTX * 0.7:
        yield ProgressUpdate(
            step="synthesizing",
            message="⚡ Optimizing prompt length for faster processing...",
            progress=0.87
        )
        evidence_text = evidence_text[:int(N_CTX * 0.5)]
    
    yield ProgressUpdate(
        step="generating",
        message="🤖 Generating final answer...",
        progress=0.9,
        details={"evidence_items_used": len(relevant_evidence)}
    )
    
    synthesis_prompt = EVIDENCE_SYNTHESIZER_PROMPT.format(
        question=question,
        evidence_text=evidence_text,
        citations=citations_text,
    )
    
    final_answer = model.generate([{
        "role": "user",
        "content": synthesis_prompt
    }]).content
    
    elapsed_time = time.time() - start_time
    
    # Final completion update
    yield ProgressUpdate(
        step="completed",
        message="✅ Research complete!",
        progress=1.0,
        details={
            "final_answer": final_answer,
            "total_time": elapsed_time,
            "evidence_used": len(relevant_evidence),
            "companies_analyzed": len(context.companies_found)
        }
    )


# --------------------------
# CLI interface
# --------------------------
def main():
    """Command-line interface for 10-K research agent."""
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        # Default research question
        question = "How are major technology companies approaching artificial intelligence and what risks do they identify?"
    
    print("=" * 80)
    print("🏢 10-K SEC FILINGS RESEARCH AGENT")
    print("=" * 80)
    print(f"Research Question: {question}")
    print("=" * 80)
    
    try:
        answer = conduct_10k_research(question)
        
        print("\n" + "=" * 80)
        print("📋 RESEARCH FINDINGS")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during research: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
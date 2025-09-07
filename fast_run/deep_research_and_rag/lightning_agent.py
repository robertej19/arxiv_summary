#!/usr/bin/env python3
"""
Lightning 10-K Research Agent - Sub-100ms Responses

Complete runtime engine that combines question analysis, semantic search,
and instant synthesis for lightning-fast responses with proper citations.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import our components
from question_analyzer import QuestionAnalyzer, QueryPlan
from semantic_fact_db import SemanticFactDB
from instant_synthesizer import InstantSynthesizer

# --------------------------
# Performance Monitor
# --------------------------

class PerformanceMonitor:
    """Monitor and report performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timers."""
        self.times = {}
        self.start_time = None
    
    def start(self, component: str):
        """Start timing a component."""
        self.times[component] = {"start": time.time()}
    
    def end(self, component: str):
        """End timing a component."""
        if component in self.times:
            self.times[component]["end"] = time.time()
            self.times[component]["duration"] = self.times[component]["end"] - self.times[component]["start"]
    
    def get_total_time(self) -> float:
        """Get total processing time."""
        if not self.times:
            return 0.0
        
        start = min(times["start"] for times in self.times.values())
        end = max(times["end"] for times in self.times.values() if "end" in times)
        return end - start
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown by component."""
        return {
            component: times.get("duration", 0.0) * 1000  # Convert to ms
            for component, times in self.times.items()
        }
    
    def report(self, verbose: bool = True) -> str:
        """Generate performance report."""
        total_ms = self.get_total_time() * 1000
        breakdown = self.get_breakdown()
        
        if verbose:
            report = f"⚡ Total: {total_ms:.1f}ms\n"
            report += "   Breakdown:\n"
            for component, duration_ms in breakdown.items():
                percentage = (duration_ms / total_ms * 100) if total_ms > 0 else 0
                report += f"   • {component}: {duration_ms:.1f}ms ({percentage:.1f}%)\n"
        else:
            report = f"⚡ {total_ms:.1f}ms"
        
        return report

# --------------------------
# Lightning Agent
# --------------------------

class LightningAgent:
    """Ultra-fast 10-K research agent with sub-100ms response times."""
    
    def __init__(self, knowledge_store_dir: str = "knowledge_store", verbose: bool = False):
        self.verbose = verbose
        self.monitor = PerformanceMonitor()
        
        if self.verbose:
            print("🚀 Initializing Lightning Agent...")
        
        # Initialize components
        self.monitor.start("initialization")
        
        try:
            self.analyzer = QuestionAnalyzer()
            self.db = SemanticFactDB(knowledge_store_dir)
            self.synthesizer = InstantSynthesizer(knowledge_store_dir)
            
            self.monitor.end("initialization")
            
            if self.verbose:
                init_time = self.monitor.get_breakdown()["initialization"]
                print(f"✅ Lightning Agent ready! Initialization: {init_time:.1f}ms")
                self._print_stats()
        
        except Exception as e:
            print(f"❌ Failed to initialize Lightning Agent: {e}")
            raise
    
    def answer(self, question: str, detailed_timing: bool = False) -> str:
        """Generate lightning-fast answer to question."""
        self.monitor.reset()
        total_start = time.time()
        
        try:
            # Step 1: Analyze question
            self.monitor.start("analysis")
            plan = self.analyzer.analyze(question)
            self.monitor.end("analysis")
            
            # Step 2: Synthesize response
            self.monitor.start("synthesis")
            response = self.synthesizer.synthesize(plan)
            self.monitor.end("synthesis")
            
            # Calculate total time
            total_time = (time.time() - total_start) * 1000
            
            # Add performance footer if requested
            if detailed_timing:
                response += f"\n\n{self.monitor.report(verbose=True)}"
            elif self.verbose:
                print(self.monitor.report(verbose=False))
            
            return response
            
        except Exception as e:
            error_time = (time.time() - total_start) * 1000
            print(f"❌ Error processing question in {error_time:.1f}ms: {e}")
            return f"Error processing question: {str(e)}"
    
    def batch_answer(self, questions: list[str]) -> Dict[str, Any]:
        """Process multiple questions and return batch results."""
        results = {
            "questions": [],
            "answers": [],
            "timings": [],
            "total_time": 0,
            "average_time": 0
        }
        
        batch_start = time.time()
        
        for question in questions:
            start = time.time()
            answer = self.answer(question)
            duration = (time.time() - start) * 1000
            
            results["questions"].append(question)
            results["answers"].append(answer)
            results["timings"].append(duration)
        
        results["total_time"] = (time.time() - batch_start) * 1000
        results["average_time"] = sum(results["timings"]) / len(results["timings"]) if results["timings"] else 0
        
        return results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and stats."""
        db_stats = self.db.get_database_stats()
        
        return {
            "name": "Lightning 10-K Research Agent",
            "description": "Sub-100ms responses from 10-K SEC filings",
            "capabilities": [
                "Instant factual lookup",
                "Company comparisons", 
                "Trend analysis",
                "Risk assessment",
                "Strategy analysis",
                "Industry insights"
            ],
            "database_stats": db_stats,
            "supported_query_types": [
                "What is [company]'s revenue?",
                "Compare [company1] vs [company2]",
                "How has [metric] changed over time?",
                "What are the main risks in [sector]?",
                "What is [company]'s AI strategy?"
            ],
            "performance_target": "< 100ms response time",
            "citation_format": "Numbered references to 10-K filings"
        }
    
    def _print_stats(self):
        """Print database statistics."""
        stats = self.db.get_database_stats()
        print(f"📊 Database: {stats['total_facts']:,} facts, {stats['companies_covered']} companies")
        print(f"📅 Years: {stats['years_covered'][0]}-{stats['years_covered'][-1]}")
        print(f"🔝 Top topics: {', '.join([topic for topic, _ in stats['top_topics'][:3]])}")

# --------------------------
# Preprocessing Pipeline
# --------------------------

def build_knowledge_base(db_path: str = "10k_knowledge_base.db", 
                        output_dir: str = "knowledge_store") -> bool:
    """Complete preprocessing pipeline to build knowledge base."""
    
    print("=" * 70)
    print("🏗️  LIGHTNING AGENT PREPROCESSING PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Extract knowledge from 10-K corpus
        print("\n📚 Step 1: Knowledge Extraction")
        print("-" * 40)
        
        from knowledge_extractor import CorpusProcessor
        processor = CorpusProcessor(db_path)
        knowledge_graph = processor.process_corpus(output_dir)
        
        # Step 2: Build semantic database
        print("\n🧠 Step 2: Semantic Database Creation")
        print("-" * 40)
        
        from semantic_fact_db import SemanticFactDB
        semantic_db = SemanticFactDB(output_dir)
        
        # Verify everything is working
        print("\n✅ Step 3: Verification")
        print("-" * 40)
        
        stats = semantic_db.get_database_stats()
        print(f"   Total facts: {stats['total_facts']:,}")
        print(f"   Companies: {stats['companies_covered']}")
        print(f"   Years covered: {stats['years_covered'][0]}-{stats['years_covered'][-1]}")
        
        # Test performance
        test_queries = [
            "Apple revenue growth",
            "Microsoft AI strategy", 
            "cybersecurity risks"
        ]
        
        print("\n⚡ Performance Test:")
        for query in test_queries:
            start = time.time()
            results = semantic_db.quick_search(query, max_results=3)
            elapsed = (time.time() - start) * 1000
            print(f"   '{query}': {len(results)} results in {elapsed:.1f}ms")
        
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE - Ready for lightning-fast queries!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# --------------------------
# CLI Interface
# --------------------------

def main():
    """Command-line interface for Lightning Agent."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Lightning 10-K Research Agent")
    parser.add_argument("question", nargs="*", help="Research question")
    parser.add_argument("--build", action="store_true", help="Build knowledge base from corpus")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--timing", "-t", action="store_true", help="Show detailed timing")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    # Build knowledge base if requested
    if args.build:
        success = build_knowledge_base()
        return 0 if success else 1
    
    # Check if knowledge base exists
    knowledge_store = Path("knowledge_store")
    if not knowledge_store.exists() or not (knowledge_store / "semantic_db.pkl").exists():
        print("❌ Knowledge base not found. Run with --build first:")
        print("   python lightning_agent.py --build")
        return 1
    
    try:
        # Initialize agent
        agent = LightningAgent(verbose=args.verbose)
        
        # Show statistics if requested
        if args.stats:
            capabilities = agent.get_capabilities()
            stats = capabilities["database_stats"]
            print("\n📊 DATABASE STATISTICS:")
            print(f"   Facts: {stats['total_facts']:,}")
            print(f"   Companies: {stats['companies_covered']}")
            print(f"   Years: {stats['years_covered'][0]}-{stats['years_covered'][-1]}")
            print(f"   Top topics: {', '.join([t[0] for t in stats['top_topics'][:5]])}")
            return 0
        
        # Run benchmark if requested
        if args.benchmark:
            print("\n⚡ LIGHTNING AGENT BENCHMARK")
            print("=" * 50)
            
            benchmark_questions = [
                "What is Apple's revenue in 2023?",
                "Compare Microsoft and Google AI strategies",
                "How has Tesla's employee count changed?",
                "What cybersecurity risks do tech companies face?",
                "What is Amazon's supply chain strategy?",
                "How do AAPL and MSFT compare financially?",
                "What are the main regulatory risks?",
                "Show me revenue trends for major tech companies"
            ]
            
            results = agent.batch_answer(benchmark_questions)
            
            print(f"\n📊 BENCHMARK RESULTS:")
            print(f"   Questions processed: {len(results['questions'])}")
            print(f"   Total time: {results['total_time']:.1f}ms")
            print(f"   Average time: {results['average_time']:.1f}ms")
            print(f"   Min time: {min(results['timings']):.1f}ms")
            print(f"   Max time: {max(results['timings']):.1f}ms")
            
            # Show sub-100ms success rate
            sub_100ms = sum(1 for t in results['timings'] if t < 100)
            success_rate = (sub_100ms / len(results['timings'])) * 100
            print(f"   Sub-100ms success rate: {success_rate:.1f}% ({sub_100ms}/{len(results['timings'])})")
            
            return 0
        
        # Interactive mode
        if args.interactive:
            print("\n🚀 LIGHTNING AGENT - INTERACTIVE MODE")
            print("=" * 50)
            print("Ask questions about 10-K filings. Type 'quit' to exit.")
            print("Example: 'What is Apple's AI strategy?'")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\n❓ Question: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not question:
                        continue
                    
                    answer = agent.answer(question, detailed_timing=args.timing)
                    print(f"\n💬 Answer:\n{answer}")
                    print("-" * 50)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("\n👋 Thanks for using Lightning Agent!")
            return 0
        
        # Single question mode
        if args.question:
            question = " ".join(args.question)
            print(f"\n❓ Question: {question}")
            print("=" * 70)
            
            answer = agent.answer(question, detailed_timing=args.timing)
            print(f"\n💬 Answer:\n{answer}")
            print("=" * 70)
            
            return 0
        
        # Show usage if no arguments
        parser.print_help()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

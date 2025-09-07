#!/usr/bin/env python3
"""
Instant Answer Synthesis Engine

Combines facts from semantic database into coherent, citable responses
in sub-100ms using template-based synthesis and fact aggregation.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from question_analyzer import QueryPlan, QueryIntent, QueryScope
from semantic_fact_db import SemanticFactDB, SearchResult, SemanticQuery
from knowledge_extractor import Fact

# --------------------------
# Response Templates
# --------------------------

class ResponseTemplates:
    """Template-based response generation for common query patterns."""
    
    COMPARISON_TEMPLATE = """Based on 10-K filings, here's how {company1} and {company2} compare on {topic}:

**{company1}**: {company1_facts}

**{company2}**: {company2_facts}

{synthesis}

Sources: {citations}"""

    TREND_TEMPLATE = """Based on {company}'s 10-K filings from {years}, here's the trend for {topic}:

{trend_analysis}

{key_insights}

Sources: {citations}"""

    FACTUAL_TEMPLATE = """Based on {company}'s recent 10-K filings:

{main_facts}

{additional_context}

Sources: {citations}"""

    RISK_TEMPLATE = """Key {topic} risks identified in recent 10-K filings:

{risk_points}

{risk_summary}

Sources: {citations}"""

    STRATEGY_TEMPLATE = """Based on 10-K filings, here's how companies are approaching {topic}:

{strategy_points}

{strategic_insights}

Sources: {citations}"""

    INDUSTRY_TEMPLATE = """Analysis of {topic} across the {industry} sector based on 10-K filings:

{industry_overview}

{key_trends}

{notable_companies}

Sources: {citations}"""

# --------------------------
# Fact Aggregator
# --------------------------

class FactAggregator:
    """Aggregates and organizes facts for synthesis."""
    
    def __init__(self):
        pass
    
    def group_by_company(self, facts: List[Fact]) -> Dict[str, List[Fact]]:
        """Group facts by company."""
        grouped = defaultdict(list)
        for fact in facts:
            grouped[fact.source_company].append(fact)
        return dict(grouped)
    
    def group_by_topic(self, facts: List[Fact]) -> Dict[str, List[Fact]]:
        """Group facts by topic."""
        grouped = defaultdict(list)
        for fact in facts:
            for topic in fact.topics:
                grouped[topic].append(fact)
        return dict(grouped)
    
    def group_by_year(self, facts: List[Fact]) -> Dict[int, List[Fact]]:
        """Group facts by year."""
        grouped = defaultdict(list)
        for fact in facts:
            grouped[fact.source_year].append(fact)
        return dict(grouped)
    
    def find_temporal_patterns(self, facts: List[Fact]) -> Dict[str, Any]:
        """Analyze temporal patterns in facts."""
        yearly_facts = self.group_by_year(facts)
        
        if len(yearly_facts) < 2:
            return {"has_trend": False}
        
        years = sorted(yearly_facts.keys())
        pattern_analysis = {
            "has_trend": True,
            "year_range": f"{years[0]}-{years[-1]}",
            "total_years": len(years),
            "yearly_counts": {year: len(yearly_facts[year]) for year in years}
        }
        
        # Look for increasing/decreasing patterns
        counts = [len(yearly_facts[year]) for year in years]
        if len(counts) >= 3:
            if counts[-1] > counts[0] and counts[-1] > counts[1]:
                pattern_analysis["trend"] = "increasing"
            elif counts[-1] < counts[0] and counts[-1] < counts[1]:
                pattern_analysis["trend"] = "decreasing"
            else:
                pattern_analysis["trend"] = "stable"
        
        return pattern_analysis
    
    def extract_key_metrics(self, facts: List[Fact]) -> Dict[str, List[str]]:
        """Extract key metrics and values from facts."""
        metrics = defaultdict(list)
        
        for fact in facts:
            if "revenue" in fact.predicate.lower():
                metrics["revenue"].append(fact.object)
            elif "profit" in fact.predicate.lower() or "income" in fact.predicate.lower():
                metrics["profit"].append(fact.object)
            elif "employee" in fact.predicate.lower():
                metrics["employees"].append(fact.object)
            elif "risk" in fact.predicate.lower():
                metrics["risks"].append(fact.object)
            elif "strategy" in fact.predicate.lower():
                metrics["strategy"].append(fact.object)
        
        return dict(metrics)

# --------------------------
# Citation Manager
# --------------------------

class CitationManager:
    """Manages citations and source tracking."""
    
    def __init__(self):
        self.citation_counter = 1
        self.fact_to_citation = {}
    
    def add_fact(self, fact: Fact) -> str:
        """Add fact and return citation reference."""
        if fact.id not in self.fact_to_citation:
            citation_num = self.citation_counter
            self.fact_to_citation[fact.id] = citation_num
            self.citation_counter += 1
        
        return f"[{self.fact_to_citation[fact.id]}]"
    
    def generate_bibliography(self, facts: List[Fact]) -> str:
        """Generate bibliography from facts."""
        citations = []
        
        # Group by source document for cleaner citations
        by_source = defaultdict(list)
        for fact in facts:
            if fact.id in self.fact_to_citation:
                citation_num = self.fact_to_citation[fact.id]
                key = f"{fact.source_company}_{fact.source_year}"
                by_source[key].append((citation_num, fact))
        
        for source_key, fact_list in by_source.items():
            fact_list.sort(key=lambda x: x[0])  # Sort by citation number
            first_fact = fact_list[0][1]
            citation_nums = [str(x[0]) for x in fact_list]
            
            if len(citation_nums) == 1:
                num_str = citation_nums[0]
            else:
                num_str = f"{citation_nums[0]}-{citation_nums[-1]}"
            
            citation = f"[{num_str}] {first_fact.source_company} 10-K Filing, FY{first_fact.source_year}"
            citations.append(citation)
        
        return "\n".join(citations)

# --------------------------
# Content Synthesizer
# --------------------------

class ContentSynthesizer:
    """Synthesizes coherent responses from facts."""
    
    def __init__(self):
        self.templates = ResponseTemplates()
        self.aggregator = FactAggregator()
        self.citation_manager = CitationManager()
    
    def synthesize_comparison(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize comparison between companies."""
        if len(plan.companies) < 2:
            return self._fallback_synthesis(facts)
        
        company_facts = self.aggregator.group_by_company(facts)
        company1, company2 = plan.companies[0], plan.companies[1]
        
        facts1 = company_facts.get(company1, [])
        facts2 = company_facts.get(company2, [])
        
        if not facts1 or not facts2:
            return self._fallback_synthesis(facts)
        
        # Generate company-specific summaries
        summary1 = self._summarize_company_facts(facts1, company1)
        summary2 = self._summarize_company_facts(facts2, company2)
        
        # Generate synthesis
        topic = plan.topics[0] if plan.topics else "business performance"
        synthesis = self._generate_comparison_synthesis(facts1, facts2, topic)
        
        # Generate citations
        all_facts = facts1 + facts2
        citations = self.citation_manager.generate_bibliography(all_facts)
        
        return self.templates.COMPARISON_TEMPLATE.format(
            company1=company1,
            company2=company2,
            topic=topic,
            company1_facts=summary1,
            company2_facts=summary2,
            synthesis=synthesis,
            citations=citations
        )
    
    def synthesize_trend(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize temporal trend analysis."""
        if not plan.companies:
            return self._fallback_synthesis(facts)
        
        company = plan.companies[0]
        temporal_analysis = self.aggregator.find_temporal_patterns(facts)
        
        if not temporal_analysis.get("has_trend"):
            return self._fallback_synthesis(facts)
        
        # Generate trend analysis
        trend_text = self._generate_trend_analysis(facts, temporal_analysis)
        insights = self._generate_trend_insights(facts, temporal_analysis)
        
        topic = plan.topics[0] if plan.topics else "business metrics"
        citations = self.citation_manager.generate_bibliography(facts)
        
        return self.templates.TREND_TEMPLATE.format(
            company=company,
            years=temporal_analysis["year_range"],
            topic=topic,
            trend_analysis=trend_text,
            key_insights=insights,
            citations=citations
        )
    
    def synthesize_factual(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize factual response."""
        if not facts:
            return "No relevant information found in the 10-K database."
        
        # Group facts by relevance and importance
        primary_facts = facts[:3]  # Most relevant
        supporting_facts = facts[3:6] if len(facts) > 3 else []
        
        main_text = self._generate_factual_summary(primary_facts)
        context_text = self._generate_supporting_context(supporting_facts) if supporting_facts else ""
        
        company = plan.companies[0] if plan.companies else "companies"
        citations = self.citation_manager.generate_bibliography(facts)
        
        return self.templates.FACTUAL_TEMPLATE.format(
            company=company,
            main_facts=main_text,
            additional_context=context_text,
            citations=citations
        )
    
    def synthesize_risk(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize risk analysis."""
        risk_facts = [f for f in facts if "risk" in " ".join(f.topics)]
        
        if not risk_facts:
            return self._fallback_synthesis(facts)
        
        # Group by risk type
        risk_groups = self.aggregator.group_by_topic(risk_facts)
        risk_points = self._generate_risk_points(risk_groups)
        risk_summary = self._generate_risk_summary(risk_facts)
        
        topic = "risk" if not plan.topics else plan.topics[0]
        citations = self.citation_manager.generate_bibliography(risk_facts)
        
        return self.templates.RISK_TEMPLATE.format(
            topic=topic,
            risk_points=risk_points,
            risk_summary=risk_summary,
            citations=citations
        )
    
    def synthesize_strategy(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize strategy analysis."""
        strategy_facts = [f for f in facts if "strategy" in " ".join(f.topics)]
        
        if not strategy_facts:
            return self._fallback_synthesis(facts)
        
        strategy_points = self._generate_strategy_points(strategy_facts)
        insights = self._generate_strategy_insights(strategy_facts)
        
        topic = plan.topics[0] if plan.topics else "business strategy"
        citations = self.citation_manager.generate_bibliography(strategy_facts)
        
        return self.templates.STRATEGY_TEMPLATE.format(
            topic=topic,
            strategy_points=strategy_points,
            strategic_insights=insights,
            citations=citations
        )
    
    def synthesize_industry(self, plan: QueryPlan, facts: List[Fact]) -> str:
        """Synthesize industry-wide analysis."""
        company_groups = self.aggregator.group_by_company(facts)
        
        overview = self._generate_industry_overview(company_groups)
        trends = self._generate_industry_trends(facts)
        notable = self._generate_notable_companies(company_groups)
        
        topic = plan.topics[0] if plan.topics else "business trends"
        industry = "technology"  # Could be enhanced to detect industry
        citations = self.citation_manager.generate_bibliography(facts)
        
        return self.templates.INDUSTRY_TEMPLATE.format(
            topic=topic,
            industry=industry,
            industry_overview=overview,
            key_trends=trends,
            notable_companies=notable,
            citations=citations
        )
    
    def _fallback_synthesis(self, facts: List[Fact]) -> str:
        """Fallback synthesis for unclassified queries."""
        if not facts:
            return "No relevant information found in the 10-K database."
        
        # Simple fact-by-fact presentation
        response_parts = []
        for i, fact in enumerate(facts[:5], 1):
            citation = self.citation_manager.add_fact(fact)
            text = f"• {fact.source_company} ({fact.source_year}): {fact.object[:200]}... {citation}"
            response_parts.append(text)
        
        citations = self.citation_manager.generate_bibliography(facts[:5])
        
        return "\n".join(response_parts) + f"\n\nSources:\n{citations}"
    
    # Helper methods for content generation
    def _summarize_company_facts(self, facts: List[Fact], company: str) -> str:
        """Summarize facts for a single company."""
        if not facts:
            return f"Limited information available for {company}."
        
        key_points = []
        for fact in facts[:3]:  # Top 3 facts
            citation = self.citation_manager.add_fact(fact)
            point = f"{fact.object[:150]}... {citation}"
            key_points.append(point)
        
        return "; ".join(key_points)
    
    def _generate_comparison_synthesis(self, facts1: List[Fact], facts2: List[Fact], topic: str) -> str:
        """Generate synthesis comparing two companies."""
        # Simple comparison logic
        if len(facts1) > len(facts2):
            return f"Based on available data, {facts1[0].source_company} has more extensive disclosures about {topic}."
        elif len(facts2) > len(facts1):
            return f"Based on available data, {facts2[0].source_company} has more extensive disclosures about {topic}."
        else:
            return f"Both companies provide similar levels of disclosure about {topic}."
    
    def _generate_trend_analysis(self, facts: List[Fact], temporal: Dict[str, Any]) -> str:
        """Generate trend analysis text."""
        trend = temporal.get("trend", "stable")
        years = temporal.get("total_years", 0)
        
        if trend == "increasing":
            return f"The data shows an increasing trend over {years} years, with more disclosures in recent periods."
        elif trend == "decreasing":
            return f"The data shows a decreasing trend over {years} years, with fewer disclosures in recent periods."
        else:
            return f"The data shows a stable pattern over {years} years, with consistent disclosure levels."
    
    def _generate_trend_insights(self, facts: List[Fact], temporal: Dict[str, Any]) -> str:
        """Generate trend insights."""
        return "The trend analysis suggests evolving business priorities and regulatory focus areas."
    
    def _generate_factual_summary(self, facts: List[Fact]) -> str:
        """Generate summary from primary facts."""
        summaries = []
        for fact in facts:
            citation = self.citation_manager.add_fact(fact)
            summary = f"• {fact.object[:200]}... {citation}"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _generate_supporting_context(self, facts: List[Fact]) -> str:
        """Generate supporting context from additional facts."""
        if not facts:
            return ""
        
        return f"\nAdditional context from {len(facts)} related disclosures provides further details on this topic."
    
    def _generate_risk_points(self, risk_groups: Dict[str, List[Fact]]) -> str:
        """Generate risk point summaries."""
        points = []
        for risk_type, risk_facts in risk_groups.items():
            if risk_facts:
                citation = self.citation_manager.add_fact(risk_facts[0])
                point = f"**{risk_type.title()}**: {risk_facts[0].object[:150]}... {citation}"
                points.append(point)
        
        return "\n\n".join(points)
    
    def _generate_risk_summary(self, facts: List[Fact]) -> str:
        """Generate overall risk summary."""
        companies = set(fact.source_company for fact in facts)
        return f"These risks are consistently identified across {len(companies)} companies in recent filings."
    
    def _generate_strategy_points(self, facts: List[Fact]) -> str:
        """Generate strategy point summaries."""
        points = []
        for fact in facts[:5]:
            citation = self.citation_manager.add_fact(fact)
            point = f"• **{fact.source_company}**: {fact.object[:150]}... {citation}"
            points.append(point)
        
        return "\n\n".join(points)
    
    def _generate_strategy_insights(self, facts: List[Fact]) -> str:
        """Generate strategic insights."""
        companies = set(fact.source_company for fact in facts)
        return f"Strategic approaches vary across {len(companies)} companies, reflecting different market positions and priorities."
    
    def _generate_industry_overview(self, company_groups: Dict[str, List[Fact]]) -> str:
        """Generate industry overview."""
        return f"Analysis based on disclosures from {len(company_groups)} companies shows diverse approaches and priorities."
    
    def _generate_industry_trends(self, facts: List[Fact]) -> str:
        """Generate industry trend analysis."""
        return "Common themes emerge across multiple companies, indicating industry-wide trends and challenges."
    
    def _generate_notable_companies(self, company_groups: Dict[str, List[Fact]]) -> str:
        """Generate notable companies section."""
        # Sort by number of relevant facts
        sorted_companies = sorted(company_groups.items(), key=lambda x: len(x[1]), reverse=True)
        top_companies = [company for company, _ in sorted_companies[:3]]
        
        return f"Companies with the most relevant disclosures: {', '.join(top_companies)}"

# --------------------------
# Main Instant Synthesizer
# --------------------------

class InstantSynthesizer:
    """Main synthesis engine that coordinates all components."""
    
    def __init__(self, knowledge_store_dir: str = "knowledge_store"):
        self.db = SemanticFactDB(knowledge_store_dir)
        self.synthesizer = ContentSynthesizer()
    
    def synthesize(self, plan: QueryPlan) -> str:
        """Synthesize instant response from query plan."""
        start_time = time.time()
        
        # Create semantic query from plan
        semantic_query = self._plan_to_semantic_query(plan)
        
        # Search for relevant facts
        search_results = self.db.search(semantic_query)
        
        if not search_results:
            return "No relevant information found in the 10-K database for this question."
        
        # Extract facts from search results
        facts = [result.fact for result in search_results]
        
        # Choose synthesis method based on intent
        if plan.intent == QueryIntent.COMPARISON:
            response = self.synthesizer.synthesize_comparison(plan, facts)
        elif plan.intent == QueryIntent.TREND:
            response = self.synthesizer.synthesize_trend(plan, facts)
        elif plan.intent == QueryIntent.RISK_ANALYSIS:
            response = self.synthesizer.synthesize_risk(plan, facts)
        elif plan.intent == QueryIntent.STRATEGY:
            response = self.synthesizer.synthesize_strategy(plan, facts)
        elif plan.scope == QueryScope.INDUSTRY:
            response = self.synthesizer.synthesize_industry(plan, facts)
        else:
            response = self.synthesizer.synthesize_factual(plan, facts)
        
        # Apply length constraints
        if len(response) > plan.max_response_length:
            response = response[:plan.max_response_length - 50] + "... [truncated]"
        
        synthesis_time = time.time() - start_time
        print(f"💬 Response synthesized in {synthesis_time*1000:.1f}ms - {len(facts)} facts used")
        
        return response
    
    def _plan_to_semantic_query(self, plan: QueryPlan) -> SemanticQuery:
        """Convert query plan to semantic query."""
        return SemanticQuery(
            text=plan.original_question,
            entity_filter=[],
            company_filter=plan.companies,
            year_filter=plan.years,
            topic_filter=plan.topics,
            fact_types=plan.fact_types,
            max_results=15  # Get enough facts for good synthesis
        )

# --------------------------
# CLI Interface
# --------------------------

def main():
    """Test instant synthesis."""
    print("=" * 70)
    print("💬 INSTANT SYNTHESIS ENGINE TEST")
    print("=" * 70)
    
    from question_analyzer import QuestionAnalyzer
    
    try:
        analyzer = QuestionAnalyzer()
        synthesizer = InstantSynthesizer()
        
        test_questions = [
            "How has Apple's revenue grown?",
            "Compare Microsoft and Google AI strategies",
            "What cybersecurity risks do tech companies face?",
            "How many employees does Tesla have?",
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 50)
            
            # Analyze question
            plan = analyzer.analyze(question)
            
            # Synthesize response
            response = synthesizer.synthesize(plan)
            
            print(f"💬 Response:\n{response}")
            print("=" * 70)
        
        print("✅ Synthesis testing complete!")
        
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

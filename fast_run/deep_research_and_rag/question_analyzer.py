#!/usr/bin/env python3
"""
Question Analysis and Query Planning

Real-time question classification, entity extraction, and query planning
for instant synthesis from semantic fact database.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import spacy
from collections import Counter

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spacy model: python -m spacy download en_core_web_sm")
    import sys
    sys.exit(1)

# --------------------------
# Query Intent Classification
# --------------------------

class QueryIntent(Enum):
    """Types of query intents we can handle."""
    COMPARISON = "comparison"          # "Compare Apple vs Microsoft"
    TREND = "trend"                   # "How has revenue changed over time"
    FACTUAL = "factual"              # "What is Apple's revenue"
    RISK_ANALYSIS = "risk"           # "What are cybersecurity risks"
    STRATEGY = "strategy"            # "What is Apple's AI strategy"
    FINANCIAL = "financial"          # "Show me financial metrics"
    OPERATIONAL = "operational"      # "How many employees does Apple have"
    REGULATORY = "regulatory"        # "What regulatory issues does Apple face"
    UNKNOWN = "unknown"              # Cannot classify

class QueryScope(Enum):
    """Scope of the query."""
    SINGLE_COMPANY = "single_company"    # About one specific company
    MULTI_COMPANY = "multi_company"     # About multiple companies
    INDUSTRY = "industry"               # About entire industry/sector
    CROSS_SECTOR = "cross_sector"       # Across multiple sectors

@dataclass
class QueryPlan:
    """Complete analysis and execution plan for a query."""
    original_question: str
    intent: QueryIntent
    scope: QueryScope
    
    # Extracted entities
    companies: List[str]           # Company tickers/names
    years: List[int]              # Specific years mentioned
    metrics: List[str]            # Financial/operational metrics
    topics: List[str]             # Business topics (AI, risks, etc.)
    entities: List[str]           # Other named entities
    
    # Query execution hints
    fact_types: List[str]         # Types of facts to prioritize
    requires_aggregation: bool    # Whether answer needs aggregation
    requires_comparison: bool     # Whether answer needs comparison
    requires_temporal: bool       # Whether answer needs time analysis
    confidence: float             # Confidence in classification
    
    # Response formatting hints
    preferred_format: str         # "paragraph", "bullet_points", "table"
    max_response_length: int      # Suggested response length

# --------------------------
# Entity Extractors
# --------------------------

class CompanyExtractor:
    """Extract company names and tickers from text."""
    
    def __init__(self):
        # Common company name patterns and their tickers
        self.company_patterns = {
            r'\b(apple|aapl)\b': 'AAPL',
            r'\b(microsoft|msft)\b': 'MSFT', 
            r'\b(google|alphabet|googl|goog)\b': 'GOOGL',
            r'\b(amazon|amzn)\b': 'AMZN',
            r'\b(meta|facebook|fb)\b': 'META',
            r'\b(tesla|tsla)\b': 'TSLA',
            r'\b(nvidia|nvda)\b': 'NVDA',
            r'\b(intel|intc)\b': 'INTC',
            r'\b(oracle|orcl)\b': 'ORCL',
            r'\b(salesforce|crm)\b': 'CRM',
            r'\b(adobe|adbe)\b': 'ADBE',
            r'\b(netflix|nflx)\b': 'NFLX',
            r'\b(uber|uber)\b': 'UBER',
            r'\b(zoom|zm)\b': 'ZM',
            r'\b(palantir|pltr)\b': 'PLTR'
        }
        
        # Common ticker pattern
        self.ticker_pattern = r'\b([A-Z]{2,5})\b'
    
    def extract(self, text: str) -> List[str]:
        """Extract company tickers from text."""
        text_lower = text.lower()
        companies = set()
        
        # Check company name patterns
        for pattern, ticker in self.company_patterns.items():
            if re.search(pattern, text_lower):
                companies.add(ticker)
        
        # Check for direct ticker mentions
        ticker_matches = re.findall(self.ticker_pattern, text.upper())
        for ticker in ticker_matches:
            # Filter out common false positives
            if ticker not in ['THE', 'AND', 'FOR', 'ARE', 'ALL', 'LLC', 'INC', 'LTD', 'USA', 'CEO', 'CFO']:
                companies.add(ticker)
        
        return list(companies)

class MetricExtractor:
    """Extract financial and operational metrics from text."""
    
    def __init__(self):
        self.metric_patterns = {
            'revenue': r'\b(revenue|sales|income|earnings|turnover)\b',
            'profit': r'\b(profit|net income|earnings|margin)\b',
            'employees': r'\b(employees|workforce|staff|headcount)\b',
            'market_cap': r'\b(market cap|market capitalization|valuation)\b',
            'debt': r'\b(debt|liabilities|borrowing)\b',
            'cash': r'\b(cash|liquidity|reserves)\b',
            'growth': r'\b(growth|increase|expansion)\b',
            'costs': r'\b(costs|expenses|spending)\b',
            'investment': r'\b(investment|capex|r&d|research)\b',
            'customers': r'\b(customers|users|subscribers)\b'
        }
    
    def extract(self, text: str) -> List[str]:
        """Extract metrics mentioned in text."""
        text_lower = text.lower()
        metrics = []
        
        for metric, pattern in self.metric_patterns.items():
            if re.search(pattern, text_lower):
                metrics.append(metric)
        
        return metrics

class TopicExtractor:
    """Extract business topics and themes from text."""
    
    def __init__(self):
        self.topic_patterns = {
            'ai': r'\b(ai|artificial intelligence|machine learning|automation|neural|deep learning)\b',
            'cybersecurity': r'\b(cyber|security|breach|hack|privacy|data protection)\b',
            'risks': r'\b(risk|threat|challenge|concern|uncertainty|vulnerability)\b',
            'strategy': r'\b(strategy|strategic|vision|roadmap|plan|initiative)\b',
            'supply_chain': r'\b(supply chain|logistics|supplier|vendor|procurement)\b',
            'sustainability': r'\b(sustainability|green|environmental|carbon|climate|renewable)\b',
            'regulation': r'\b(regulation|regulatory|compliance|government|policy|legal)\b',
            'competition': r'\b(competition|competitive|competitor|rival|market share)\b',
            'innovation': r'\b(innovation|research|development|technology|breakthrough)\b',
            'digital': r'\b(digital|cloud|software|platform|online|internet)\b'
        }
    
    def extract(self, text: str) -> List[str]:
        """Extract topics mentioned in text."""
        text_lower = text.lower()
        topics = []
        
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, text_lower):
                topics.append(topic)
        
        return topics

class YearExtractor:
    """Extract years from text."""
    
    def __init__(self):
        self.year_patterns = [
            r'\b(20[0-9]{2})\b',      # 2000-2099
            r'\b(19[89][0-9])\b',     # 1980-1999
            r'\bfy\s*(20[0-9]{2})\b', # FY2023
            r'\bfiscal\s*year\s*(20[0-9]{2})\b'
        ]
    
    def extract(self, text: str) -> List[int]:
        """Extract years mentioned in text."""
        years = set()
        
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                year = int(match)
                if 1990 <= year <= 2030:  # Reasonable range
                    years.add(year)
        
        return sorted(list(years))

# --------------------------
# Intent Classifier
# --------------------------

class IntentClassifier:
    """Classify query intent based on patterns and keywords."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.COMPARISON: [
                r'\b(compare|versus|vs|difference|against|between)\b',
                r'\b(which is better|how do they differ|what are the differences)\b'
            ],
            QueryIntent.TREND: [
                r'\b(trend|over time|change|evolution|growth|decline|trajectory)\b',
                r'\b(how has|what happened|increased|decreased|improved|worsened)\b',
                r'\b(year over year|quarterly|annually|historical)\b'
            ],
            QueryIntent.RISK_ANALYSIS: [
                r'\b(risk|threat|challenge|concern|problem|issue|vulnerability)\b',
                r'\b(what could go wrong|potential problems|main concerns)\b'
            ],
            QueryIntent.STRATEGY: [
                r'\b(strategy|strategic|approach|plan|vision|roadmap|initiative)\b',
                r'\b(how do they plan|what is their approach|business model)\b'
            ],
            QueryIntent.FINANCIAL: [
                r'\b(revenue|profit|income|earnings|financial|money|cash|debt)\b',
                r'\b(how much|financial performance|fiscal|quarterly results)\b'
            ],
            QueryIntent.OPERATIONAL: [
                r'\b(employees|operations|manufacturing|production|workforce)\b',
                r'\b(how many|organizational|operational efficiency)\b'
            ],
            QueryIntent.REGULATORY: [
                r'\b(regulation|regulatory|compliance|government|policy|legal)\b',
                r'\b(sec filings|regulatory requirements|government oversight)\b'
            ]
        }
    
    def classify(self, text: str) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score."""
        text_lower = text.lower()
        
        intent_scores = Counter()
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            top_intent = intent_scores.most_common(1)[0]
            confidence = min(top_intent[1] / 5.0, 1.0)  # Normalize to 0-1
            return top_intent[0], confidence
        
        return QueryIntent.FACTUAL, 0.5  # Default to factual with medium confidence

# --------------------------
# Scope Analyzer
# --------------------------

class ScopeAnalyzer:
    """Determine the scope of the query."""
    
    def analyze(self, companies: List[str], topics: List[str], text: str) -> QueryScope:
        """Determine query scope based on entities and text."""
        text_lower = text.lower()
        
        # Check for explicit scope indicators
        if re.search(r'\b(industry|sector|market|all companies|companies in general)\b', text_lower):
            return QueryScope.INDUSTRY
        
        if re.search(r'\b(across sectors|multiple industries|different sectors)\b', text_lower):
            return QueryScope.CROSS_SECTOR
        
        # Based on number of companies mentioned
        if len(companies) == 0:
            return QueryScope.INDUSTRY  # General question
        elif len(companies) == 1:
            return QueryScope.SINGLE_COMPANY
        else:
            return QueryScope.MULTI_COMPANY

# --------------------------
# Main Question Analyzer
# --------------------------

class QuestionAnalyzer:
    """Main analyzer that orchestrates all components."""
    
    def __init__(self):
        self.company_extractor = CompanyExtractor()
        self.metric_extractor = MetricExtractor()
        self.topic_extractor = TopicExtractor()
        self.year_extractor = YearExtractor()
        self.intent_classifier = IntentClassifier()
        self.scope_analyzer = ScopeAnalyzer()
    
    def analyze(self, question: str) -> QueryPlan:
        """Analyze question and create comprehensive query plan."""
        start_time = time.time()
        
        # Extract entities
        companies = self.company_extractor.extract(question)
        metrics = self.metric_extractor.extract(question)
        topics = self.topic_extractor.extract(question)
        years = self.year_extractor.extract(question)
        
        # Extract general named entities using spaCy
        doc = nlp(question)
        entities = [ent.text.strip() for ent in doc.ents 
                   if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']]
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify(question)
        
        # Determine scope
        scope = self.scope_analyzer.analyze(companies, topics, question)
        
        # Determine execution hints
        requires_aggregation = len(companies) > 1 or scope in [QueryScope.INDUSTRY, QueryScope.CROSS_SECTOR]
        requires_comparison = intent == QueryIntent.COMPARISON or len(companies) > 1
        requires_temporal = intent == QueryIntent.TREND or len(years) > 1
        
        # Determine fact types to prioritize
        fact_types = self._determine_fact_types(intent, metrics, topics)
        
        # Determine response format
        preferred_format = self._determine_response_format(intent, scope, len(companies))
        max_response_length = self._determine_response_length(intent, scope)
        
        analysis_time = time.time() - start_time
        
        plan = QueryPlan(
            original_question=question,
            intent=intent,
            scope=scope,
            companies=companies,
            years=years,
            metrics=metrics,
            topics=topics,
            entities=entities,
            fact_types=fact_types,
            requires_aggregation=requires_aggregation,
            requires_comparison=requires_comparison,
            requires_temporal=requires_temporal,
            confidence=confidence,
            preferred_format=preferred_format,
            max_response_length=max_response_length
        )
        
        print(f"🧠 Question analyzed in {analysis_time*1000:.1f}ms - Intent: {intent.value}, Scope: {scope.value}")
        
        return plan
    
    def _determine_fact_types(self, intent: QueryIntent, metrics: List[str], topics: List[str]) -> List[str]:
        """Determine which fact types to prioritize."""
        fact_types = []
        
        if intent == QueryIntent.FINANCIAL or 'revenue' in metrics or 'profit' in metrics:
            fact_types.extend(['has_revenue', 'has_net_income', 'has_profit'])
        
        if intent == QueryIntent.RISK_ANALYSIS or 'risks' in topics:
            fact_types.extend(['identifies_cybersecurity_risk', 'identifies_regulatory_risk', 
                             'identifies_competition_risk'])
        
        if intent == QueryIntent.STRATEGY or 'strategy' in topics:
            fact_types.extend(['strategy_focus_ai', 'strategy_focus_sustainability'])
        
        if 'employees' in metrics:
            fact_types.append('has_employees')
        
        return fact_types
    
    def _determine_response_format(self, intent: QueryIntent, scope: QueryScope, num_companies: int) -> str:
        """Determine preferred response format."""
        if intent == QueryIntent.COMPARISON or num_companies > 1:
            return "bullet_points"
        elif scope == QueryScope.INDUSTRY:
            return "paragraph"
        else:
            return "paragraph"
    
    def _determine_response_length(self, intent: QueryIntent, scope: QueryScope) -> int:
        """Determine suggested response length in characters."""
        if scope == QueryScope.INDUSTRY:
            return 800  # Longer for industry overviews
        elif intent == QueryIntent.COMPARISON:
            return 600  # Medium for comparisons
        else:
            return 400  # Shorter for specific queries

# --------------------------
# CLI Interface
# --------------------------

def main():
    """Test question analysis."""
    print("=" * 70)
    print("🧠 QUESTION ANALYZER TEST")
    print("=" * 70)
    
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "How has Apple's revenue grown over the past 3 years?",
        "Compare Microsoft and Google's AI strategies",
        "What are the main cybersecurity risks facing tech companies?",
        "How many employees does Tesla have?",
        "What regulatory challenges is Meta facing?",
        "Show me Amazon's financial performance in 2023",
        "What are the biggest risks in the technology sector?",
        "How do AAPL and MSFT compare in terms of innovation?",
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        plan = analyzer.analyze(question)
        
        print(f"   Intent: {plan.intent.value} (confidence: {plan.confidence:.2f})")
        print(f"   Scope: {plan.scope.value}")
        print(f"   Companies: {plan.companies}")
        print(f"   Topics: {plan.topics}")
        print(f"   Metrics: {plan.metrics}")
        print(f"   Years: {plan.years}")
        print(f"   Format: {plan.preferred_format}")
        print("-" * 50)
    
    print("\n✅ Question analysis complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Knowledge Extraction Pipeline for 10-K Filings

Extracts facts, entities, relationships, and builds comprehensive knowledge graph
from 10-K corpus for instant synthesis.
"""

from __future__ import annotations

import re
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict, Counter

# Load spacy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spacy model: python -m spacy download en_core_web_sm")
    import sys
    sys.exit(1)

# --------------------------
# Core Data Structures
# --------------------------

@dataclass
class Fact:
    """A single extractable fact from 10-K filings."""
    id: str                    # Unique fact ID
    subject: str              # What the fact is about (company, metric, etc.)
    predicate: str            # Relationship/property type
    object: str               # Value/description
    source_company: str       # Company ticker
    source_year: int          # Fiscal year
    source_section: str       # 10-K section
    source_file: str          # Original file path
    confidence: float         # Extraction confidence (0-1)
    context: str              # Surrounding text for verification
    topics: List[str]         # Associated topics/themes
    entities: List[str]       # Named entities in fact
    
    def __post_init__(self):
        if not self.id:
            # Generate deterministic ID from content
            content = f"{self.subject}_{self.predicate}_{self.object}_{self.source_company}_{self.source_year}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass 
class Entity:
    """A named entity extracted from corpus."""
    name: str                 # Entity name
    type: str                # Entity type (COMPANY, PRODUCT, LOCATION, etc.)
    mentions: int             # Number of mentions across corpus
    companies: Set[str]       # Companies that mention this entity
    years: Set[int]          # Years when mentioned
    contexts: List[str]      # Sample contexts where mentioned
    aliases: Set[str]        # Alternative names/spellings

@dataclass
class Topic:
    """A semantic topic cluster."""
    name: str                # Topic name
    keywords: List[str]      # Key terms associated with topic
    companies: Set[str]      # Companies discussing this topic
    facts: List[str]         # Fact IDs related to this topic
    subtopics: List[str]     # Sub-topic hierarchy
    importance: float        # Topic importance score (based on frequency)

@dataclass
class KnowledgeGraph:
    """Complete knowledge graph of the corpus."""
    facts: Dict[str, Fact] = field(default_factory=dict)
    entities: Dict[str, Entity] = field(default_factory=dict)
    topics: Dict[str, Topic] = field(default_factory=dict)
    companies: Dict[str, Dict] = field(default_factory=dict)
    
    # Indices for fast lookup
    facts_by_company: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    facts_by_topic: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    facts_by_year: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    facts_by_predicate: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

# --------------------------
# Text Processing & Extraction
# --------------------------

class TextProcessor:
    """Processes 10-K text and extracts structured information."""
    
    def __init__(self):
        self.financial_patterns = {
            'revenue': r'(?:revenue|net sales|total revenue)\s*(?:of|was|:)?\s*\$?([0-9,\.]+)\s*(?:billion|million|thousand|B|M|K)?',
            'net_income': r'(?:net income|net earnings|profit)\s*(?:of|was|:)?\s*\$?([0-9,\.]+)\s*(?:billion|million|thousand|B|M|K)?',
            'employees': r'(?:employ|workforce|team|staff)(?:ed|s)?\s*(?:approximately|about|over|around)?\s*([0-9,]+)',
            'market_cap': r'market\s*(?:capitalization|cap)\s*(?:of|was|:)?\s*\$?([0-9,\.]+)\s*(?:billion|million|trillion|B|M|T)?'
        }
        
        self.risk_patterns = {
            'cybersecurity': r'(?:cyber\s*security|data\s*breach|security\s*threat|hack|malware)',
            'regulatory': r'(?:regulation|regulatory|compliance|government\s*oversight)',
            'competition': r'(?:compet\w+|rival|market\s*share|industry\s*pressure)',
            'supply_chain': r'(?:supply\s*chain|supplier|vendor|logistics|disruption)',
            'economic': r'(?:economic|recession|inflation|interest\s*rate|market\s*condition)',
            'climate': r'(?:climate|environmental|carbon|sustainability|green)'
        }
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities using spaCy."""
        doc = nlp(text[:1000000])  # Limit text length for performance
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'MONEY', 'PERCENT']:
                entities.append((ent.text.strip(), ent.label_))
        
        return entities
    
    def extract_financial_facts(self, text: str, company: str, year: int, section: str) -> List[Fact]:
        """Extract financial metrics and facts."""
        facts = []
        
        for metric, pattern in self.financial_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                context = text[max(0, match.start()-100):match.end()+100]
                
                fact = Fact(
                    id="",  # Will be auto-generated
                    subject=company,
                    predicate=f"has_{metric}_{year}",
                    object=value,
                    source_company=company,
                    source_year=year,
                    source_section=section,
                    source_file="",  # Will be set by caller
                    confidence=0.8,
                    context=context.strip(),
                    topics=['financial', metric],
                    entities=[company]
                )
                facts.append(fact)
        
        return facts
    
    def extract_risk_facts(self, text: str, company: str, year: int, section: str) -> List[Fact]:
        """Extract risk-related facts."""
        facts = []
        
        # Split text into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            for risk_type, pattern in self.risk_patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    fact = Fact(
                        id="",
                        subject=company,
                        predicate=f"identifies_{risk_type}_risk",
                        object=sentence.strip()[:200],  # Limit object length
                        source_company=company,
                        source_year=year,
                        source_section=section,
                        source_file="",
                        confidence=0.7,
                        context=sentence.strip(),
                        topics=['risks', risk_type],
                        entities=self._extract_simple_entities(sentence)
                    )
                    facts.append(fact)
        
        return facts
    
    def extract_strategy_facts(self, text: str, company: str, year: int, section: str) -> List[Fact]:
        """Extract strategic initiatives and business focus areas."""
        facts = []
        strategy_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'automation',
            'sustainability', 'renewable energy', 'carbon neutral',
            'digital transformation', 'cloud computing', 'innovation',
            'research and development', 'r&d', 'investment'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in strategy_keywords:
                if keyword in sentence_lower and len(sentence.strip()) > 30:
                    fact = Fact(
                        id="",
                        subject=company,
                        predicate=f"strategy_focus_{keyword.replace(' ', '_')}",
                        object=sentence.strip()[:300],
                        source_company=company,
                        source_year=year,
                        source_section=section,
                        source_file="",
                        confidence=0.6,
                        context=sentence.strip(),
                        topics=['strategy', keyword.replace(' ', '_')],
                        entities=self._extract_simple_entities(sentence)
                    )
                    facts.append(fact)
                    break  # Only one strategy fact per sentence
        
        return facts
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Simple entity extraction for shorter texts."""
        # Look for capitalized words that might be entities
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        return [e for e in entities if len(e) > 2 and e not in ['The', 'We', 'Our', 'This', 'These']]

# --------------------------
# Knowledge Graph Builder
# --------------------------

class KnowledgeGraphBuilder:
    """Builds comprehensive knowledge graph from extracted facts."""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.processor = TextProcessor()
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge graph and update indices."""
        self.kg.facts[fact.id] = fact
        
        # Update indices
        self.kg.facts_by_company[fact.source_company].append(fact.id)
        self.kg.facts_by_year[fact.source_year].append(fact.id)
        self.kg.facts_by_predicate[fact.predicate].append(fact.id)
        
        for topic in fact.topics:
            self.kg.facts_by_topic[topic].append(fact.id)
    
    def add_entity(self, name: str, entity_type: str, company: str, year: int, context: str):
        """Add or update an entity in the knowledge graph."""
        if name not in self.kg.entities:
            self.kg.entities[name] = Entity(
                name=name,
                type=entity_type,
                mentions=0,
                companies=set(),
                years=set(),
                contexts=[],
                aliases=set()
            )
        
        entity = self.kg.entities[name]
        entity.mentions += 1
        entity.companies.add(company)
        entity.years.add(year)
        
        if len(entity.contexts) < 10:  # Limit context storage
            entity.contexts.append(context[:200])
    
    def build_topics(self):
        """Build topic clusters from extracted facts."""
        topic_facts = defaultdict(list)
        topic_keywords = defaultdict(Counter)
        topic_companies = defaultdict(set)
        
        # Group facts by topics
        for fact_id, fact in self.kg.facts.items():
            for topic in fact.topics:
                topic_facts[topic].append(fact_id)
                topic_companies[topic].add(fact.source_company)
                
                # Extract keywords from fact content
                words = re.findall(r'\b[a-zA-Z]{3,}\b', 
                                 f"{fact.subject} {fact.predicate} {fact.object}".lower())
                for word in words:
                    if word not in ['the', 'and', 'or', 'but', 'for', 'with', 'has', 'was', 'were']:
                        topic_keywords[topic][word] += 1
        
        # Create topic objects
        for topic_name, fact_ids in topic_facts.items():
            keywords = [word for word, count in topic_keywords[topic_name].most_common(10)]
            
            self.kg.topics[topic_name] = Topic(
                name=topic_name,
                keywords=keywords,
                companies=topic_companies[topic_name],
                facts=fact_ids,
                subtopics=[],  # Could be enhanced with hierarchical clustering
                importance=len(fact_ids)  # Simple importance based on fact count
            )
    
    def build_company_profiles(self):
        """Build comprehensive company profiles."""
        for company_ticker in set(fact.source_company for fact in self.kg.facts.values()):
            company_facts = [self.kg.facts[fid] for fid in self.kg.facts_by_company[company_ticker]]
            
            # Extract key metrics
            years = sorted(set(fact.source_year for fact in company_facts))
            topics = Counter()
            for fact in company_facts:
                topics.update(fact.topics)
            
            self.kg.companies[company_ticker] = {
                'ticker': company_ticker,
                'years_covered': years,
                'total_facts': len(company_facts),
                'top_topics': dict(topics.most_common(10)),
                'sections_covered': set(fact.source_section for fact in company_facts),
                'fact_ids': [fact.id for fact in company_facts]
            }

# --------------------------
# Corpus Processor
# --------------------------

class CorpusProcessor:
    """Main processor for building knowledge graph from 10-K corpus."""
    
    def __init__(self, db_path: str = "10k_knowledge_base.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            # Try parent directory
            parent_db = Path(__file__).parent.parent / db_path
            if parent_db.exists():
                self.db_path = parent_db
            else:
                raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.builder = KnowledgeGraphBuilder()
        
    def process_corpus(self, output_dir: str = "knowledge_store") -> KnowledgeGraph:
        """Process entire 10-K corpus and build knowledge graph."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🏗️  Building knowledge graph from 10-K corpus...")
        
        conn = duckdb.connect(str(self.db_path))
        try:
            # Get all sections from database
            sql = """
                SELECT 
                    f.ticker,
                    f.company_name,
                    f.fiscal_year,
                    f.file_path,
                    s.section_name,
                    s.content
                FROM sections s
                JOIN filings f ON s.filing_id = f.id
                ORDER BY f.ticker, f.fiscal_year, s.section_name
            """
            
            results = conn.execute(sql).fetchall()
            print(f"📊 Processing {len(results)} sections from corpus...")
            
            processed_count = 0
            for row in results:
                ticker, company_name, fiscal_year, file_path, section_name, content = row
                
                if content and len(content.strip()) > 100:  # Skip empty sections
                    self._process_section(ticker, company_name, fiscal_year, 
                                        file_path, section_name, content)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"   Processed {processed_count} sections...")
            
            print(f"✅ Processed {processed_count} sections")
            
        finally:
            conn.close()
        
        # Build topic clusters and company profiles
        print("🔗 Building topic clusters...")
        self.builder.build_topics()
        
        print("🏢 Building company profiles...")
        self.builder.build_company_profiles()
        
        # Save knowledge graph
        kg_file = output_path / "knowledge_graph.pkl"
        with open(kg_file, 'wb') as f:
            pickle.dump(self.builder.kg, f)
        
        print(f"💾 Knowledge graph saved to {kg_file}")
        print(f"📈 Stats: {len(self.builder.kg.facts)} facts, {len(self.builder.kg.entities)} entities, {len(self.builder.kg.topics)} topics")
        
        return self.builder.kg
    
    def _process_section(self, ticker: str, company_name: str, fiscal_year: int,
                        file_path: str, section_name: str, content: str):
        """Process a single 10-K section and extract facts."""
        
        # Extract different types of facts based on section
        facts = []
        
        if section_name.lower() in ['business', 'overview', 'operations']:
            facts.extend(self.builder.processor.extract_financial_facts(
                content, ticker, fiscal_year, section_name))
            facts.extend(self.builder.processor.extract_strategy_facts(
                content, ticker, fiscal_year, section_name))
        
        elif 'risk' in section_name.lower():
            facts.extend(self.builder.processor.extract_risk_facts(
                content, ticker, fiscal_year, section_name))
        
        else:
            # For other sections, extract general facts
            facts.extend(self.builder.processor.extract_financial_facts(
                content, ticker, fiscal_year, section_name))
        
        # Set file path for all facts
        for fact in facts:
            fact.source_file = file_path
            self.builder.add_fact(fact)
        
        # Extract and add entities
        entities = self.builder.processor.extract_entities(content)
        for entity_name, entity_type in entities:
            if len(entity_name) > 2 and entity_name != ticker:  # Skip company's own ticker
                self.builder.add_entity(entity_name, entity_type, ticker, 
                                      fiscal_year, content[:200])

# --------------------------
# CLI Interface
# --------------------------

def main():
    """Build knowledge graph from 10-K corpus."""
    import sys
    
    print("=" * 70)
    print("🧠 10-K KNOWLEDGE EXTRACTION PIPELINE")
    print("=" * 70)
    
    try:
        processor = CorpusProcessor()
        knowledge_graph = processor.process_corpus()
        
        print("\n🎯 KNOWLEDGE EXTRACTION COMPLETE!")
        print("=" * 70)
        print(f"📊 Total Facts: {len(knowledge_graph.facts):,}")
        print(f"🏷️  Total Entities: {len(knowledge_graph.entities):,}")
        print(f"📂 Total Topics: {len(knowledge_graph.topics):,}")
        print(f"🏢 Companies Covered: {len(knowledge_graph.companies):,}")
        
        # Show top topics
        if knowledge_graph.topics:
            print(f"\n🔝 Top Topics by Fact Count:")
            sorted_topics = sorted(knowledge_graph.topics.items(), 
                                 key=lambda x: x[1].importance, reverse=True)
            for i, (topic_name, topic) in enumerate(sorted_topics[:10], 1):
                print(f"   {i:2d}. {topic_name}: {len(topic.facts)} facts")
        
        print("=" * 70)
        print("✅ Ready for instant synthesis! Run semantic_fact_db.py next.")
        
    except Exception as e:
        print(f"❌ Error during knowledge extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

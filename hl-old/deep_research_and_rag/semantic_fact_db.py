#!/usr/bin/env python3
"""
Semantic Fact Database for Instant Retrieval

Builds semantic embeddings and FAISS indices for sub-100ms fact retrieval
from extracted knowledge graph.
"""

from __future__ import annotations

import pickle
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from knowledge_extractor import KnowledgeGraph, Fact, Entity, Topic

# --------------------------
# Configuration
# --------------------------

# Use lightweight, fast embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, fast inference
EMBEDDING_DIM = 384
BATCH_SIZE = 32  # For efficient embedding generation

# Index configurations
INDEX_CONFIG = {
    'facts': {'index_type': 'flatip', 'nprobe': 10},
    'entities': {'index_type': 'flatip', 'nprobe': 5},
    'topics': {'index_type': 'flatip', 'nprobe': 5}
}

# --------------------------
# Semantic Database
# --------------------------

@dataclass
class SemanticQuery:
    """Query structure for semantic search."""
    text: str                    # Query text
    entity_filter: List[str]     # Filter by entities
    company_filter: List[str]    # Filter by companies  
    year_filter: List[int]       # Filter by years
    topic_filter: List[str]      # Filter by topics
    fact_types: List[str]        # Filter by fact predicates
    max_results: int = 10        # Maximum results to return

@dataclass
class SearchResult:
    """Search result with fact and relevance score."""
    fact: Fact
    score: float                 # Semantic similarity score
    rank: int                    # Result ranking

class SemanticFactDB:
    """High-performance semantic database for instant fact retrieval."""
    
    def __init__(self, knowledge_store_dir: str = "knowledge_store"):
        self.store_dir = Path(knowledge_store_dir)
        self.kg: Optional[KnowledgeGraph] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Embedding matrices and indices
        self.fact_embeddings: Optional[np.ndarray] = None
        self.fact_index: Optional[faiss.Index] = None
        self.fact_ids: List[str] = []
        
        self.entity_embeddings: Optional[np.ndarray] = None
        self.entity_index: Optional[faiss.Index] = None
        self.entity_names: List[str] = []
        
        self.topic_embeddings: Optional[np.ndarray] = None
        self.topic_index: Optional[faiss.Index] = None
        self.topic_names: List[str] = []
        
        # Lookup tables for fast filtering
        self.fact_lookup: Dict[str, int] = {}  # fact_id -> index
        self.company_facts: Dict[str, List[int]] = {}  # company -> fact indices
        self.year_facts: Dict[int, List[int]] = {}  # year -> fact indices
        self.topic_facts: Dict[str, List[int]] = {}  # topic -> fact indices
        
        self._load_or_build()
    
    def _load_or_build(self):
        """Load existing database or build from knowledge graph."""
        embeddings_file = self.store_dir / "semantic_db.pkl"
        
        if embeddings_file.exists():
            print("📚 Loading existing semantic database...")
            self._load_database(embeddings_file)
        else:
            print("🏗️  Building semantic database from knowledge graph...")
            self._build_database()
            self._save_database(embeddings_file)
    
    def _load_database(self, embeddings_file: Path):
        """Load pre-built semantic database."""
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.kg = data['knowledge_graph']
        self.fact_embeddings = data['fact_embeddings']
        self.fact_ids = data['fact_ids']
        self.entity_embeddings = data['entity_embeddings']
        self.entity_names = data['entity_names']
        self.topic_embeddings = data['topic_embeddings']
        self.topic_names = data['topic_names']
        self.fact_lookup = data['fact_lookup']
        self.company_facts = data['company_facts']
        self.year_facts = data['year_facts']
        self.topic_facts = data['topic_facts']
        
        # Rebuild FAISS indices
        self._build_indices()
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        print(f"✅ Loaded semantic database: {len(self.fact_ids)} facts, {len(self.entity_names)} entities")
    
    def _build_database(self):
        """Build semantic database from knowledge graph."""
        # Load knowledge graph
        kg_file = self.store_dir / "knowledge_graph.pkl"
        if not kg_file.exists():
            raise FileNotFoundError(f"Knowledge graph not found at {kg_file}. Run knowledge_extractor.py first.")
        
        with open(kg_file, 'rb') as f:
            self.kg = pickle.load(f)
        
        print(f"📊 Processing {len(self.kg.facts)} facts for semantic indexing...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        
        # Build fact embeddings
        self._build_fact_embeddings()
        
        # Build entity embeddings  
        self._build_entity_embeddings()
        
        # Build topic embeddings
        self._build_topic_embeddings()
        
        # Build lookup tables
        self._build_lookup_tables()
        
        # Build FAISS indices
        self._build_indices()
    
    def _build_fact_embeddings(self):
        """Build embeddings for all facts."""
        print("🔤 Building fact embeddings...")
        
        facts_list = list(self.kg.facts.values())
        self.fact_ids = [fact.id for fact in facts_list]
        
        # Create text representations for embedding
        fact_texts = []
        for fact in facts_list:
            # Combine multiple fields for richer representation
            text = f"{fact.subject} {fact.predicate} {fact.object}"
            if fact.context:
                text += f" {fact.context[:100]}"  # Add context snippet
            fact_texts.append(text)
        
        # Generate embeddings in batches
        self.fact_embeddings = self._embed_texts(fact_texts)
        print(f"✅ Generated {self.fact_embeddings.shape[0]} fact embeddings")
    
    def _build_entity_embeddings(self):
        """Build embeddings for entities."""
        print("🏷️  Building entity embeddings...")
        
        self.entity_names = list(self.kg.entities.keys())
        
        # Create entity text representations
        entity_texts = []
        for entity_name in self.entity_names:
            entity = self.kg.entities[entity_name]
            # Combine entity name with type and sample contexts
            text = f"{entity.name} {entity.type}"
            if entity.contexts:
                text += f" {entity.contexts[0][:50]}"  # Add sample context
            entity_texts.append(text)
        
        self.entity_embeddings = self._embed_texts(entity_texts)
        print(f"✅ Generated {self.entity_embeddings.shape[0]} entity embeddings")
    
    def _build_topic_embeddings(self):
        """Build embeddings for topics."""
        print("📂 Building topic embeddings...")
        
        self.topic_names = list(self.kg.topics.keys())
        
        # Create topic text representations
        topic_texts = []
        for topic_name in self.topic_names:
            topic = self.kg.topics[topic_name]
            # Combine topic name with keywords
            text = f"{topic.name} {' '.join(topic.keywords[:10])}"
            topic_texts.append(text)
        
        self.topic_embeddings = self._embed_texts(topic_texts)
        print(f"✅ Generated {self.topic_embeddings.shape[0]} topic embeddings")
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts."""
        embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings).astype(np.float32)
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast filtering."""
        print("🔍 Building lookup tables...")
        
        # Fact ID to index mapping
        self.fact_lookup = {fact_id: i for i, fact_id in enumerate(self.fact_ids)}
        
        # Company-based lookup
        self.company_facts = {}
        for company, fact_ids in self.kg.facts_by_company.items():
            indices = [self.fact_lookup[fid] for fid in fact_ids if fid in self.fact_lookup]
            self.company_facts[company] = indices
        
        # Year-based lookup
        self.year_facts = {}
        for year, fact_ids in self.kg.facts_by_year.items():
            indices = [self.fact_lookup[fid] for fid in fact_ids if fid in self.fact_lookup]
            self.year_facts[year] = indices
        
        # Topic-based lookup
        self.topic_facts = {}
        for topic, fact_ids in self.kg.facts_by_topic.items():
            indices = [self.fact_lookup[fid] for fid in fact_ids if fid in self.fact_lookup]
            self.topic_facts[topic] = indices
        
        print("✅ Built lookup tables for fast filtering")
    
    def _build_indices(self):
        """Build FAISS indices for fast similarity search."""
        print("⚡ Building FAISS indices...")
        
        # Fact index
        self.fact_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.fact_index.add(self.fact_embeddings)
        
        # Entity index
        self.entity_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.entity_index.add(self.entity_embeddings)
        
        # Topic index
        self.topic_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.topic_index.add(self.topic_embeddings)
        
        print("✅ Built FAISS indices for instant search")
    
    def _save_database(self, embeddings_file: Path):
        """Save semantic database to disk."""
        print("💾 Saving semantic database...")
        
        data = {
            'knowledge_graph': self.kg,
            'fact_embeddings': self.fact_embeddings,
            'fact_ids': self.fact_ids,
            'entity_embeddings': self.entity_embeddings,
            'entity_names': self.entity_names,
            'topic_embeddings': self.topic_embeddings,
            'topic_names': self.topic_names,
            'fact_lookup': self.fact_lookup,
            'company_facts': self.company_facts,
            'year_facts': self.year_facts,
            'topic_facts': self.topic_facts
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved semantic database to {embeddings_file}")
    
    # --------------------------
    # Search Interface
    # --------------------------
    
    def search(self, query: SemanticQuery) -> List[SearchResult]:
        """Perform semantic search with filtering."""
        start_time = time.time()
        
        # Generate query embedding
        query_emb = self.embedding_model.encode([query.text], normalize_embeddings=True)
        
        # Find candidate facts using semantic similarity
        distances, indices = self.fact_index.search(query_emb.astype(np.float32), k=min(1000, len(self.fact_ids)))
        
        # Apply filters
        filtered_results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= len(self.fact_ids):
                continue
                
            fact_id = self.fact_ids[idx]
            fact = self.kg.facts[fact_id]
            
            # Apply filters
            if query.company_filter and fact.source_company not in query.company_filter:
                continue
            if query.year_filter and fact.source_year not in query.year_filter:
                continue
            if query.topic_filter and not any(topic in fact.topics for topic in query.topic_filter):
                continue
            if query.fact_types and fact.predicate not in query.fact_types:
                continue
            
            filtered_results.append(SearchResult(
                fact=fact,
                score=float(score),
                rank=i + 1
            ))
            
            if len(filtered_results) >= query.max_results:
                break
        
        search_time = time.time() - start_time
        print(f"🔍 Search completed in {search_time*1000:.1f}ms - {len(filtered_results)} results")
        
        return filtered_results
    
    def quick_search(self, text: str, max_results: int = 5) -> List[SearchResult]:
        """Quick search with minimal filtering for fastest results."""
        query = SemanticQuery(
            text=text,
            entity_filter=[],
            company_filter=[],
            year_filter=[],
            topic_filter=[],
            fact_types=[],
            max_results=max_results
        )
        return self.search(query)
    
    def search_by_company(self, text: str, companies: List[str], max_results: int = 5) -> List[SearchResult]:
        """Search facts for specific companies."""
        query = SemanticQuery(
            text=text,
            entity_filter=[],
            company_filter=companies,
            year_filter=[],
            topic_filter=[],
            fact_types=[],
            max_results=max_results
        )
        return self.search(query)
    
    def search_by_topic(self, text: str, topics: List[str], max_results: int = 5) -> List[SearchResult]:
        """Search facts within specific topics."""
        query = SemanticQuery(
            text=text,
            entity_filter=[],
            company_filter=[],
            year_filter=[],
            topic_filter=topics,
            fact_types=[],
            max_results=max_results
        )
        return self.search(query)
    
    def get_company_stats(self, company: str) -> Dict[str, Any]:
        """Get quick stats for a company."""
        if company not in self.kg.companies:
            return {}
        
        return self.kg.companies[company]
    
    def get_topic_stats(self, topic: str) -> Dict[str, Any]:
        """Get quick stats for a topic."""
        if topic not in self.kg.topics:
            return {}
        
        topic_obj = self.kg.topics[topic]
        return {
            'name': topic_obj.name,
            'keywords': topic_obj.keywords,
            'companies': list(topic_obj.companies),
            'fact_count': len(topic_obj.facts),
            'importance': topic_obj.importance
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        return {
            'total_facts': len(self.fact_ids),
            'total_entities': len(self.entity_names),
            'total_topics': len(self.topic_names),
            'companies_covered': len(self.kg.companies),
            'years_covered': sorted(self.kg.facts_by_year.keys()) if self.kg.facts_by_year else [],
            'top_topics': sorted(
                [(name, len(topic.facts)) for name, topic in self.kg.topics.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

# --------------------------
# CLI Interface  
# --------------------------

def main():
    """Build and test semantic fact database."""
    print("=" * 70)
    print("🧠 SEMANTIC FACT DATABASE BUILDER")
    print("=" * 70)
    
    try:
        # Build database
        db = SemanticFactDB()
        
        # Show stats
        stats = db.get_database_stats()
        print(f"\n📊 DATABASE STATISTICS:")
        print(f"   Facts: {stats['total_facts']:,}")
        print(f"   Entities: {stats['total_entities']:,}")
        print(f"   Topics: {stats['total_topics']:,}")
        print(f"   Companies: {stats['companies_covered']:,}")
        print(f"   Years: {min(stats['years_covered'])}-{max(stats['years_covered'])}")
        
        print(f"\n🔝 TOP TOPICS:")
        for i, (topic, count) in enumerate(stats['top_topics'][:5], 1):
            print(f"   {i}. {topic}: {count} facts")
        
        # Test search performance
        print(f"\n⚡ PERFORMANCE TEST:")
        test_queries = [
            "artificial intelligence risks",
            "revenue growth trends",
            "cybersecurity threats",
            "supply chain disruption"
        ]
        
        for query in test_queries:
            start = time.time()
            results = db.quick_search(query, max_results=3)
            elapsed = (time.time() - start) * 1000
            print(f"   '{query}': {len(results)} results in {elapsed:.1f}ms")
        
        print("=" * 70)
        print("✅ Semantic database ready for instant synthesis!")
        
    except Exception as e:
        print(f"❌ Error building semantic database: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

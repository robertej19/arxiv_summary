# local_corpus/tools_local.py
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from smolagents import Tool

# -----------------------------------------------------------------------------
# Paths & models
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
STORE_DIR = HERE / "store"

# Files produced by ingest.py & build_index.py
CHUNKS_PATH = STORE_DIR / "chunks.jsonl"
BM25_PKL = STORE_DIR / "bm25.pkl"
TOK_PKL = STORE_DIR / "tokens.pkl"
FAISS_INDEX = STORE_DIR / "faiss.index"
IDS_JSON = STORE_DIR / "chunk_ids.json"

# Models (cache once online; then export HF_HUB_OFFLINE=1)
EMB_MODEL = os.getenv("LOCAL_EMB_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-base")

# Candidate caps before reranking
TOP_K_BM25 = 200
TOP_K_VEC = 200

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _require_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}. Did you run ingest/build_index?")

def _load_indices() -> Tuple[BM25Okapi, List[List[str]], faiss.Index, List[str], Dict[str, Dict[str, Any]]]:
    """Load BM25, tokens, FAISS, id list, and build a chunk_id -> record map."""
    _require_exists(BM25_PKL, "BM25 index")
    _require_exists(TOK_PKL, "BM25 tokens")
    _require_exists(FAISS_INDEX, "FAISS index")
    _require_exists(IDS_JSON, "chunk id list")
    _require_exists(CHUNKS_PATH, "chunks.jsonl")

    with open(BM25_PKL, "rb") as f:
        bm25: BM25Okapi = pickle.load(f)
    with open(TOK_PKL, "rb") as f:
        tokens: List[List[str]] = pickle.load(f)
    with open(IDS_JSON, "r", encoding="utf-8") as f:
        chunk_ids: List[str] = json.load(f)

    faiss_index = faiss.read_index(str(FAISS_INDEX))

    # Build random-access map from chunk_id -> full record
    id2rec: Dict[str, Dict[str, Any]] = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id2rec[rec["chunk_id"]] = rec

    return bm25, tokens, faiss_index, chunk_ids, id2rec

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

class LocalSearchTool(Tool):
    """
    Hybrid local search over the RL corpus:
      - BM25 (keyword) + vector (FAISS, cosine) union
      - Cross-encoder reranker to pick the top-k
    Returns: {"results": [ {chunk_id, doc_id, title, year, score, snippet}, ... ]}
    """
    name = "local_search"
    description = "Search a local RL corpus (BM25 + vector + rerank)."
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language query.",
            "nullable": True,     # <- match signature Optional[str]
            "default": None,
        },
        "k": {
            "type": "integer",
            "description": "How many results to return (1–50).",
            "nullable": True,
            "default": 10,
            "minimum": 1,
            "maximum": 50,
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # Load indices & models once
        self._bm25, self._tokens, self._faiss, self._ids, self._id2rec = _load_indices()
        self._emb = SentenceTransformer(EMB_MODEL, device="cpu")
        self._rer = CrossEncoder(RERANK_MODEL, device="cpu")

    def forward(self, query: Optional[str] = None, k: Optional[int] = None) -> Dict[str, Any]:
        # Guard against missing/empty query
        if query is None or str(query).strip() == "":
            return {"results": []}

        k = 10 if k is None else max(1, min(int(k), 50))

        # --- BM25 candidates ---
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]

        # --- Vector candidates ---
        q_emb = self._emb.encode([query], normalize_embeddings=True)
        D, I = self._faiss.search(np.asarray(q_emb, dtype="float32"), TOP_K_VEC)
        vec_top_idx = I[0]

        # --- Union candidates ---
        cand_idx = list(set(bm25_top_idx.tolist()) | set(vec_top_idx.tolist()))
        if not cand_idx:
            return {"results": []}

        # --- Rerank with cross-encoder ---
        pairs = []
        for i in cand_idx:
            chunk_id = self._ids[i]
            text = self._id2rec[chunk_id]["text"][:2000]
            pairs.append((query, text))
        scores = self._rer.predict(pairs)

        ranked = sorted(zip(cand_idx, scores), key=lambda x: x[1], reverse=True)[:k]

        results: List[Dict[str, Any]] = []
        for idx, sc in ranked:
            chunk_id = self._ids[idx]
            rec = self._id2rec[chunk_id]
            results.append({
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "title": rec.get("title"),
                "year": rec.get("year"),
                "score": float(sc),
                "snippet": rec["text"][:400].replace("\n", " "),
            })

        return {"results": results}


# ReaderTool (drop-in replacement)
class ReaderTool(Tool):
    name = "read_chunk"
    description = "Read a chunk by chunk_id and return text plus a citation object."
    inputs = {
        "chunk_id": {
            "type": "string",
            "description": "Chunk identifier returned by local_search.",
            "nullable": True,        # <-- was False
            "default": None          # <-- add default since signature is optional
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        _, _, _, _, self._id2rec = _load_indices()

    def forward(self, chunk_id: str | None = None) -> Dict[str, Any]:
        if not chunk_id or chunk_id not in self._id2rec:
            return {
                "text": "",
                "citation": {
                    "doc_id": None,
                    "title": None,
                    "year": None,
                    "chunk_id": chunk_id,
                    "format": f"arxiv://unknown::{chunk_id}",
                    "warning": "chunk_id missing or not found",
                },
            }
        rec = self._id2rec[chunk_id]
        return {
            "text": rec["text"],
            "citation": {
                "doc_id": rec["doc_id"],
                "title": rec.get("title"),
                "year": rec.get("year"),
                "chunk_id": rec["chunk_id"],
                "format": f"arxiv://{rec['doc_id']}::{rec['chunk_id']}",
            },
        }

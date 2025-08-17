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
from smolagents import Tool

# Optional deps: sentence-transformers
# We keep robust fallbacks if models are unavailable offline.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

# -----------------------------------------------------------------------------
# Paths & models
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
STORE_DIR = HERE / "store"

# Files produced by your ingest/build steps
CHUNKS_PATH = STORE_DIR / "chunks.jsonl"
BM25_PKL = STORE_DIR / "bm25.pkl"
TOK_PKL = STORE_DIR / "tokens.pkl"
FAISS_INDEX = STORE_DIR / "faiss.index"
IDS_JSON = STORE_DIR / "chunk_ids.json"

# Models (use already-cached weights; set HF_HUB_OFFLINE=1 for no network)
EMB_MODEL = os.getenv("LOCAL_EMB_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-base")

# Candidate caps before reranking
TOP_K_BM25 = int(os.getenv("LOCAL_TOP_K_BM25", "200"))
TOP_K_VEC = int(os.getenv("LOCAL_TOP_K_VEC", "200"))

# Hard caps / safety
MAX_RESULTS = 50
MAX_SNIPPET_CHARS = 400
MAX_RETURN_TEXT_CHARS = 4000

# -----------------------------------------------------------------------------
# Loaders & helpers
# -----------------------------------------------------------------------------

def _require_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}. Did you run ingest/build_index?")

def _load_indices() -> Tuple[
    BM25Okapi,
    List[List[str]],
    faiss.Index,
    List[str],
    Dict[str, Dict[str, Any]],
    Dict[str, List[str]],
]:
    """Load BM25, tokens, FAISS, id list, build:
       - id2rec: chunk_id -> record
       - doc2chunks: doc_id -> [chunk_id,...] in the order of chunk_ids.json
    """
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

    # Consistency check (fail fast if artifacts are out of sync)
    missing = [cid for cid in chunk_ids if cid not in id2rec]
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            f"Store mismatch: {len(missing)} chunk_ids in {IDS_JSON} "
            f"are missing from {CHUNKS_PATH}. Examples: {sample} … "
            "Rebuild indices so all artifacts are aligned."
        )

    # Build doc_id -> list of chunk_ids (use the canonical order from chunk_ids.json)
    doc2chunks: Dict[str, List[str]] = {}
    for cid in chunk_ids:
        rec = id2rec[cid]
        did = rec["doc_id"]
        doc2chunks.setdefault(did, []).append(cid)

    return bm25, tokens, faiss_index, chunk_ids, id2rec, doc2chunks



_CHUNK_ID_RE = re.compile(r"^(?P<doc>[^:]+)::c(?P<idx>\d+)$")

def _chunk_sort_key(chunk_id: str) -> Tuple[str, int]:
    m = _CHUNK_ID_RE.match(chunk_id)
    if m:
        return (m.group("doc"), int(m.group("idx")))
    # Fallback: keep stable order
    parts = chunk_id.split("::")
    return (parts[0], 10**9)

def _parse_chunk_pos(chunk_id: str) -> Tuple[str, Optional[int]]:
    m = _CHUNK_ID_RE.match(chunk_id)
    if not m:
        doc = chunk_id.split("::")[0]
        return doc, None
    return m.group("doc"), int(m.group("idx"))

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

def _safe_head(s: str, n: int) -> str:
    s = s.replace("\u0000", "")
    if len(s) <= n:
        return s
    return s[:n] + " …[truncated]"

def _sentences(text: str) -> List[str]:
    # cheap sentence split; good enough for quotes
    # keeps periods in numbers/acronyms reasonably well
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [p.strip() for p in parts if p.strip()]

def _score_quote(sent: str, keywords: List[str]) -> int:
    if not keywords:
        return 0
    s = sent.lower()
    return sum(k.lower() in s for k in keywords)

# -----------------------------------------------------------------------------
# Global (lazy) state
# -----------------------------------------------------------------------------

_BM25: Optional[BM25Okapi] = None
_TOKENS: Optional[List[List[str]]] = None
_FAISS: Optional[faiss.Index] = None
_IDS: Optional[List[str]] = None
_ID2REC: Optional[Dict[str, Dict[str, Any]]] = None
_DOC2CHUNKS: Optional[Dict[str, List[str]]] = None
_EMB: Any = None
_RER: Any = None

def _ensure_state():
    global _BM25, _TOKENS, _FAISS, _IDS, _ID2REC, _DOC2CHUNKS, _EMB, _RER
    if _BM25 is None:
        _BM25, _TOKENS, _FAISS, _IDS, _ID2REC, _DOC2CHUNKS = _load_indices()
    if _EMB is None and SentenceTransformer is not None:
        try:
            _EMB = SentenceTransformer(EMB_MODEL, device="cpu")
        except Exception:
            _EMB = None
    if _RER is None and CrossEncoder is not None:
        try:
            _RER = CrossEncoder(RERANK_MODEL, device="cpu")
        except Exception:
            _RER = None

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import os, sys, time
from smolagents import Tool

class LocalSearchTool(Tool):
    """
    Hybrid local search over the corpus:
      - BM25 (keyword) + FAISS (semantic) union
      - Optional cross-encoder rerank
    Returns: {"results": [ {chunk_id, doc_id, title, year, score, snippet}, ... ]}
    """

    name = "local_search"
    description = "Search the local corpus (BM25 ∪ FAISS, optional cross-encoder rerank)."
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language query.",
            "nullable": True,
            "default": None,
        },
        "k": {
            "type": "integer",
            "description": "How many results to return (1–MAX_RESULTS).",
            "nullable": True,
            "default": 10,
            "minimum": 1,
            "maximum": 50,  # will be clamped again to MAX_RESULTS at runtime
        },
        "doc_id": {
            "type": "string",
            "description": "If provided, restrict results to this doc_id.",
            "nullable": True,
            "default": None,
        },
        "use_reranker": {
            "type": "boolean",
            "description": "Whether to apply cross-encoder reranking (default: true).",
            "nullable": True,
            "default": True,
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # Ensure the global caches exist in this module (created elsewhere).
        # Your file already has _ensure_state(); we reuse it here.
        try:
            _ensure_state()  # type: ignore  # loads _BM25, _FAISS, _IDS, _ID2REC, _EMB, _RER, etc.
        except NameError:
            raise RuntimeError(
                "tools_local._ensure_state() not found. Make sure this class lives in the same module as the caches."
            )

    def forward(
        self,
        query: Optional[str] = None,
        k: Optional[int] = None,
        doc_id: Optional[str] = None,
        use_reranker: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """Hybrid search (BM25 ∪ FAISS) with optional cross-encoder rerank and doc filter.
        Safe against store mismatches: skips missing chunk_ids instead of crashing.
        """
        if query is None or str(query).strip() == "":
            return {"results": []}

        # Live debug printing (opt-in via env)
        DEBUG = os.getenv("DR_DEBUG_SEARCH", "0") == "1"
        RERANK_BATCH = int(os.getenv("DR_DEBUG_BATCH", "64"))

        def dprint(msg: str):
            if DEBUG:
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()

        # Clamp k to limits
        MAX_K = int(globals().get("MAX_RESULTS", 50))
        k = 10 if k is None else max(1, min(int(k), MAX_K))
        use_rer = bool(use_reranker)

        # Pull cached globals from this module
        bm25 = globals()["_BM25"]
        faiss_index = globals()["_FAISS"]
        ids: List[str] = globals()["_IDS"]
        id2rec: Dict[str, Dict[str, Any]] = globals()["_ID2REC"]
        emb = globals().get("_EMB", None)
        rer = globals().get("_RER", None) if use_rer else None

        TOP_BM25 = int(globals().get("TOP_K_BM25", 200))
        TOP_VEC = int(globals().get("TOP_K_VEC", 200))

        t0 = time.perf_counter()
        dprint(f"[search] query={query!r} (k={k}, rerank={'on' if rer is not None else 'off'})")

        # --- BM25 candidates ---
        dprint(f"[search] BM25: scoring {len(ids)} chunks…")
        bm25_scores = bm25.get_scores(_tokenize(query))  # type: ignore
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:TOP_BM25]
        t_bm25 = time.perf_counter()
        dprint(f"[search] BM25: top{TOP_BM25} in {t_bm25 - t0:.2f}s")

        # --- Vector candidates (optional) ---
        if emb is not None:
            dprint(f"[search] FAISS: encoding & searching top{TOP_VEC}…")
            q_emb = emb.encode([query], normalize_embeddings=True, show_progress_bar=False)
            _D, I = faiss_index.search(np.asarray(q_emb, dtype="float32"), TOP_VEC)
            vec_top_idx = I[0]
            t_vec = time.perf_counter()
            dprint(f"[search] FAISS: got {len(vec_top_idx)} in {t_vec - t_bm25:.2f}s")
        else:
            vec_top_idx = np.array([], dtype=int)
            dprint("[search] FAISS: encoder unavailable, skipping vector search")

        # --- Union candidates ---
        cand_idx = list(set(bm25_top_idx.tolist()) | set(vec_top_idx.tolist()))
        if not cand_idx:
            return {"results": []}
        dprint(f"[search] union candidates: {len(cand_idx)}")

        # --- Optional doc filter (safe lookups) ---
        if doc_id:
            kept = []
            for i in cand_idx:
                cid = ids[i]
                rec = id2rec.get(cid)
                if rec is not None and rec.get("doc_id") == doc_id:
                    kept.append(i)
            cand_idx = kept
            dprint(f"[search] after doc_id filter={doc_id!r}: {len(cand_idx)} candidates")
            if not cand_idx:
                return {"results": []}

        # --- Build pairs for rerank (SAFE: skip missing) ---
        pairs: List[Tuple[str, str]] = []
        valid_idx: List[int] = []
        missing = 0
        for i in cand_idx:
            cid = ids[i]
            rec = id2rec.get(cid)
            if rec is None:
                missing += 1
                if DEBUG:
                    dprint(f"[search] WARNING: missing chunk_id {cid}; skipping")
                continue
            text = (rec.get("text") or "")[:2000]
            if not text:
                continue
            pairs.append((query, text))
            valid_idx.append(i)

        if not pairs:
            dprint("[search] no valid pairs after skipping missing/empty records")
            return {"results": []}

        # --- Rank candidates ---
        if rer is not None:
            # Cross-encoder rerank in batches -> progress prints
            dprint(f"[search] rerank: {len(pairs)} pairs (batch={RERANK_BATCH})")
            scores = np.empty(len(pairs), dtype=np.float32)
            r0 = time.perf_counter()
            for start in range(0, len(pairs), RERANK_BATCH):
                end = min(start + RERANK_BATCH, len(pairs))
                scores[start:end] = rer.predict(pairs[start:end])  # type: ignore
                if DEBUG:
                    done = end
                    pct = (done * 100) // max(1, len(pairs))
                    dprint(f"[search] rerank progress: {done}/{len(pairs)} ({pct}%)")
            dprint(f"[search] rerank done in {time.perf_counter() - r0:.2f}s (total {time.perf_counter() - t0:.2f}s)")
            scored = list(zip(valid_idx, scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            # Fallback: rank by BM25 score only (no extra compute)
            dprint("[search] rerank: disabled -> using BM25 scores")
            scored = [(i, float(bm25_scores[i])) for i in valid_idx]
            scored.sort(key=lambda x: x[1], reverse=True)

        # --- Assemble top-k results (SAFE lookups) ---
        results: List[Dict[str, Any]] = []
        for i, sc in scored[:k]:
            cid = ids[i]
            rec = id2rec.get(cid)
            if rec is None:
                continue
            results.append({
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "title": rec.get("title"),
                "year": rec.get("year"),
                "score": float(sc),
                "snippet": (rec.get("text") or "")[:400].replace("\n", " "),
            })

        if DEBUG:
            dbg_top = [(r["doc_id"], r["chunk_id"], round(r["score"], 3)) for r in results]
            dprint(f"[search] top{k}: {dbg_top}")

        return {"results": results}


class ReadChunkTool(Tool):
    name = "read_chunk"
    description = "Read a chunk by chunk_id; supports neighbor window and quote extraction."
    inputs = {
        "chunk_id": {
            "type": "string",
            "description": "Chunk identifier returned by local_search (e.g., '2112.13666v1::c5').",
            "nullable": True,  # <- required
        },
        "include_neighbors": {
            "type": "integer",
            "description": "How many neighbors to include on each side (0–3).",
            "nullable": True,
            "default": 0,
            "minimum": 0,
            "maximum": 3,
        },
        "mode": {
            "type": "string",
            "description": "Return mode: 'text' (full) or 'quotes' (extract sentences).",
            "nullable": True,
            "default": "text",
            "enum": ["text", "quotes"],
        },
        "keywords": {
            "type": "array",
            "description": "Optional keywords to guide quote selection.",
            "items": {"type": "string"},
            "nullable": True,
            "default": None,
        },
        "max_quotes": {
            "type": "integer",
            "description": "Maximum number of quotes to return when mode='quotes'.",
            "nullable": True,
            "default": 2,
            "minimum": 1,
            "maximum": 6,
        },
        "max_chars": {
            "type": "integer",
            "description": "Hard cap on returned text characters.",
            "nullable": True,
            "default": MAX_RETURN_TEXT_CHARS,
            "minimum": 500,
            "maximum": 20000,
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        _ensure_state()

    def forward(
        self,
        chunk_id: Optional[str] = None,   # <- Make it Optional to match schema
        include_neighbors: int = 0,
        mode: str = "text",
        keywords: Optional[List[str]] = None,
        max_quotes: int = 2,
        max_chars: int = MAX_RETURN_TEXT_CHARS,
    ) -> Dict[str, Any]:

        id2rec = _ID2REC  # type: ignore
        doc2chunks = _DOC2CHUNKS  # type: ignore

        if chunk_id is None:
            return {
                "text": "" if mode == "text" else [],
                "quotes": [] if mode == "quotes" else None,
                "neighbors": [],
                "citation": {
                    "doc_id": None,
                    "title": None,
                    "year": None,
                    "chunk_id": None,
                    "format": "arxiv://unknown::None",
                    "warning": "chunk_id is None",
                },
            }

        if chunk_id not in id2rec:
            return {
                "text": "" if mode == "text" else [],
                "quotes": [] if mode == "quotes" else None,
                "neighbors": [],
                "citation": {
                    "doc_id": None,
                    "title": None,
                    "year": None,
                    "chunk_id": chunk_id,
                    "format": f"arxiv://unknown::{chunk_id}",
                    "warning": "chunk_id not found",
                },
            }

        rec = id2rec[chunk_id]
        doc_id, idx = _parse_chunk_pos(chunk_id)
        neighbors: List[str] = [chunk_id]

        if include_neighbors > 0 and idx is not None:
            series = doc2chunks.get(doc_id, [])
            # gather ±N neighbors (in order)
            around = []
            for off in range(-include_neighbors, include_neighbors + 1):
                j = idx + off
                if j < 0:
                    continue
                # map idx j back to a chunk_id in series (if exists)
                # series is sorted by index, so we need to locate chunk_id with ::c{j}
                cand = f"{doc_id}::c{j}"
                if cand in id2rec:
                    around.append(cand)
            # keep unique and preserve order using series order
            seen = set()
            ordered = []
            for cid in series:
                if cid in around and cid not in seen:
                    ordered.append(cid); seen.add(cid)
            neighbors = ordered if ordered else [chunk_id]

        # Build text blob (respecting max_chars)
        texts = []
        for cid in neighbors:
            t = id2rec[cid]["text"]
            texts.append(t)
            if sum(len(x) for x in texts) >= max_chars:
                break
        full_text = _safe_head("\n\n".join(texts), max_chars)

        citation = {
            "doc_id": rec["doc_id"],
            "title": rec.get("title"),
            "year": rec.get("year"),
            "chunk_id": rec["chunk_id"],
            "format": f"arxiv://{rec['doc_id']}::{rec['chunk_id']}",
        }

        if mode == "quotes":
            kw = keywords or []
            sents = _sentences(full_text)
            # Score by keyword hits, then length (prefer 60–300 chars)
            scored = []
            for s in sents:
                if not s or len(s) < 30:  # avoid super short fragments
                    continue
                score = _score_quote(s, kw)
                # soft preference for medium length
                length_bonus = 1.0 - abs(len(s) - 180) / 300.0
                scored.append((score + max(0.0, length_bonus), s))
            scored.sort(key=lambda x: x[0], reverse=True)
            quotes = [q for _, q in scored[:max_quotes]]
            return {
                "text": None,
                "quotes": quotes,
                "neighbors": neighbors,
                "citation": citation,
            }

        # default mode: full text (bounded)
        return {
            "text": full_text,
            "quotes": None,
            "neighbors": neighbors,
            "citation": citation,
        }


# Backwards-compatible names for your imports
__all__ = [
    "LocalSearchTool",
    "ReadChunkTool",
]

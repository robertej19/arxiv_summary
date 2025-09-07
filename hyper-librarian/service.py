# service.py
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from warm_retriever import WarmRetriever
from answer_compose import select_sentences, answerability as minsource_answerability, compose_answer
from answerability_guard import assess_answerability  # NEW: stronger gate with reasons


# -------------------------
# App & config
# -------------------------
app = FastAPI(title="Hyper Librarian", version="0.2")

DUCK   = os.environ.get("HL_DUCKDB", "tenk.duckdb")
FTS    = os.environ.get("HL_FTS", "tenk_fts.sqlite")
HNSW   = os.environ.get("HL_HNSW", "tenk_hnsw.bin")
IDSNPY = os.environ.get("HL_IDS", "chunk_ids.npy")
EMBNPY = os.environ.get("HL_EMBS", "chunk_embs.npy")
MODEL  = os.environ.get("HL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval / compose knobs (can be env-driven later)
K_LEX, K_ANN, FUSE_K, K_FINAL = 80, 80, 48, 8
MMR_LAMBDA = 0.5
TARGET_SENTS = 4
MIN_SOURCES = 2   # classic guard: distinct (ticker,year,item) count

_started_at = time.time()
retr: Optional[WarmRetriever] = None


# -------------------------
# Response model
# -------------------------
class AnswerResponse(BaseModel):
    answer: str
    answerable: bool
    citations: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    timings_ms: Dict[str, float]
    reasons: Optional[List[str]] = None  # ← explain why not answerable


# -------------------------
# Lifecycle
# -------------------------
@app.on_event("startup")
def _startup() -> None:
    global retr
    retr = WarmRetriever(
        duckdb_path=DUCK,
        fts_path=FTS,
        hnsw_path=HNSW,
        ids_npy=IDSNPY,
        embs_npy=EMBNPY,
        encoder_name=MODEL,
        k_lex=K_LEX, k_ann=K_ANN, fuse_k=FUSE_K, k_final=K_FINAL, mmr_lambda=MMR_LAMBDA,
    )


# -------------------------
# Endpoints
# -------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": round(time.time() - _started_at, 1)}


@app.get("/answer", response_model=AnswerResponse)
def answer(q: str = Query(..., min_length=2, max_length=500)):
    assert retr is not None, "retriever not initialized"

    # 1) Retrieval (hybrid) → candidate chunks
    chunk_rows, timings = retr.retrieve(q)

    # 2) Sentence selection (extractive)
    picks = select_sentences(retr.duck, chunk_rows, q, target=TARGET_SENTS)

    # 3) Stronger answerability gate (returns reasons)
    ok_gate, reasons = assess_answerability(q, picks, timings, retr.sq)

    # 4) Optional: keep the classic min-sources rule too (safer together)
    ok_sources = minsource_answerability(picks, min_sources=MIN_SOURCES)
    ok = bool(ok_gate and ok_sources)

    # 5) Compose answer if answerable
    answer_text = compose_answer(picks if ok else [])

    # 6) Build payload
    citations = [
        {
            "ticker": p["ticker"],
            "year": p["year"],
            "item": p["item"],
            "doc_id": p["doc_id"],
            "section_id": p["section_id"],
            "sent_id": p["sent_id"],
            "sent_idx": p["sent_idx"],
            "hash": p["hash"],
        }
        for p in picks
    ]
    chunks = [
        {
            "chunk_id": r[0],
            "ticker": r[1],
            "year": (str(r[2])[:4] if r[2] else ""),
            "item": r[3],
            "text": " ".join((r[4] or "").split())[:320],
        }
        for r in chunk_rows
    ]

    # Include gate reasons only when not answerable
    return AnswerResponse(
        answer=answer_text,
        answerable=ok,
        citations=citations,
        chunks=chunks,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
        reasons=(None if ok else reasons),
    )

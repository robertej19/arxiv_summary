#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid (BM25 + ANN) retrieval demo with fast MMR.

Features:
- Loads precomputed chunk embeddings (chunk_ids.npy, chunk_embs.npy) if available
  for millisecond MMR; otherwise falls back to re-embedding candidates.
- Performs RRF fusion over BM25 and ANN results, then MMR for diversity.
- Prints top chunks with exact citations (ticker, filing_date, item, chunk_id, doc_id).
- Reports per-stage timings.

Usage:
  python query_demo.py --q "supply chain disruptions in 2022"

Paths (override as needed with flags):
  --duckdb tenk.duckdb
  --fts    tenk_fts.sqlite
  --hnsw   tenk_hnsw.bin
  --ids    chunk_ids.npy
  --embs   chunk_embs.npy
"""

from __future__ import annotations

import argparse
import sqlite3
import time
import re
import os
from typing import List

import duckdb
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


def rrf(ids_a: List[int], ids_b: List[int], k: int = 60, C: int = 60) -> List[int]:
    """Reciprocal Rank Fusion of two id lists."""
    score = {}
    for rank, cid in enumerate(ids_a):
        score[cid] = score.get(cid, 0.0) + 1.0 / (C + rank + 1)
    for rank, cid in enumerate(ids_b):
        score[cid] = score.get(cid, 0.0) + 1.0 / (C + rank + 1)
    return [cid for cid, _ in sorted(score.items(), key=lambda x: -x[1])][:k]


def mmr(qv: np.ndarray, cand_vecs: np.ndarray, k: int = 8, lam: float = 0.5) -> List[int]:
    """Maximal Marginal Relevance over candidate vectors (cosine; vectors assumed normalized)."""
    sims = cand_vecs @ qv  # shape (N,)
    chosen, chosen_idx, avail = [], [], list(range(len(cand_vecs)))
    if not avail:
        return chosen_idx
    # pick best by relevance
    first = int(np.argmax(sims))
    chosen.append(cand_vecs[first])
    chosen_idx.append(first)
    avail.remove(first)
    while avail and len(chosen_idx) < k:
        scores = []
        for a in avail:
            redundancy = max(float(cand_vecs[a] @ c) for c in chosen)
            score = lam * float(sims[a]) - (1.0 - lam) * redundancy
            scores.append(score)
        pick = avail[int(np.argmax(scores))]
        chosen.append(cand_vecs[pick])
        chosen_idx.append(pick)
        avail.remove(pick)
    return chosen_idx


def main():
    ap = argparse.ArgumentParser()
    # Stores
    ap.add_argument("--duckdb", default="tenk.duckdb")
    ap.add_argument("--fts", default="tenk_fts.sqlite")
    ap.add_argument("--hnsw", default="tenk_hnsw.bin")
    # Precomputed vectors (optional but strongly recommended)
    ap.add_argument("--ids", default="chunk_ids.npy")
    ap.add_argument("--embs", default="chunk_embs.npy")
    # Model
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    # Retrieval hyperparams
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k_lex", type=int, default=80)
    ap.add_argument("--k_ann", type=int, default=80)
    ap.add_argument("--k_final", type=int, default=8)
    ap.add_argument("--fuse_k", type=int, default=48, help="Candidate pool after RRF (before MMR)")
    ap.add_argument("--mmr_lambda", type=float, default=0.5)
    # Output
    ap.add_argument("--max_snippet_chars", type=int, default=320)
    args = ap.parse_args()

    t0 = time.perf_counter()

    # Load stores
    duck = duckdb.connect(args.duckdb, read_only=True)
    sq = sqlite3.connect(args.fts, check_same_thread=False)
    # Load HNSW
    # We infer dim from precomputed embs (if available); else fall back to model dim.
    use_precomputed = os.path.exists(args.ids) and os.path.exists(args.embs)
    if use_precomputed:
        ids = np.load(args.ids, mmap_mode="r")
        embs = np.load(args.embs, mmap_mode="r")
        dim = int(embs.shape[1])
    else:
        # Will be overwritten after model loads, but needed to init index safely
        dim = 384

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(args.hnsw)
    idx.set_ef(96)

    # Load model (query embedding; not used for MMR if we have precomputed vectors)
    t_model0 = time.perf_counter()
    encoder = SentenceTransformer(args.model)
    if not use_precomputed:
        # if no precomputed embs, we need true dim for the index check
        dim = encoder.get_sentence_embedding_dimension()
    t_model = (time.perf_counter() - t_model0) * 1000

    # Sanity: if embs provided, ensure dims match
    if use_precomputed and idx.dim != embs.shape[1]:
        raise SystemExit(f"HNSW dim ({idx.dim}) != precomputed embs dim ({embs.shape[1]}). Rebuild/export to match.")

    # 1) BM25 via FTS5
    t_bm0 = time.perf_counter()
    lex_rows = sq.execute(
        "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH ? LIMIT ?",
        (args.q, args.k_lex),
    ).fetchall()
    lex_ids = [int(r[0]) for r in lex_rows]
    t_bm25 = (time.perf_counter() - t_bm0) * 1000

    # 2) ANN (HNSW)
    t_ann0 = time.perf_counter()
    qv = encoder.encode([args.q], normalize_embeddings=True, convert_to_numpy=True)[0]
    labels, _ = idx.knn_query(qv, k=args.k_ann)
    ann_ids = [int(x) for x in labels[0].tolist()]
    t_ann = (time.perf_counter() - t_ann0) * 1000

    # 3) Fusion (RRF)
    t_fuse0 = time.perf_counter()
    cand_ids = rrf(lex_ids, ann_ids, k=max(args.fuse_k, args.k_final * 4))
    t_fuse = (time.perf_counter() - t_fuse0) * 1000

    # 4) MMR (diversity) — fast path with precomputed vectors
    t_mmr0 = time.perf_counter()
    if use_precomputed:
        # Map chunk_id -> row index
        id2i = {int(cid): i for i, cid in enumerate(ids)}
        # Filter to those cids we actually have vectors for
        cand_ids = [cid for cid in cand_ids if cid in id2i]
        if not cand_ids:
            final_ids = []
        else:
            cand_vecs = np.stack([embs[id2i[cid]] for cid in cand_ids], axis=0)
            picks = mmr(qv, cand_vecs, k=args.k_final, lam=args.mmr_lambda)
            final_ids = [cand_ids[i] for i in picks]
    else:
        # Fallback: re-embed candidates (slower)
        # Pull text for candidates and encode (small set)
        if cand_ids:
            placeholders = ",".join(map(str, cand_ids))
            rows = duck.execute(
                f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})"
            ).fetchall()
            # Preserve original candidate order
            id2txt = {int(cid): txt for cid, txt in rows}
            ordered_texts = [id2txt[cid] for cid in cand_ids if cid in id2txt]
            cand_vecs = encoder.encode(
                ordered_texts, normalize_embeddings=True, convert_to_numpy=True
            )
            picks = mmr(qv, cand_vecs, k=args.k_final, lam=args.mmr_lambda)
            final_ids = [cand_ids[i] for i in picks]
        else:
            final_ids = []
    t_mmr = (time.perf_counter() - t_mmr0) * 1000

    # 5) Fetch metadata for display
    t_fetch0 = time.perf_counter()
    rows = []
    if final_ids:
        placeholders = ",".join(map(str, final_ids))
        rows = duck.execute(f"""
        SELECT c.chunk_id,
               d.ticker,
               CAST(d.filing_date AS VARCHAR) AS filing_date,
               s.item,
               c.text,
               c.start_sent_idx,
               c.end_sent_idx,
               c.section_id,
               d.doc_id
        FROM chunks c
        JOIN sections s ON c.section_id = s.section_id
        JOIN docs d     ON c.doc_id     = d.doc_id
        WHERE c.chunk_id IN ({placeholders})
        """).fetchall()
    t_fetch = (time.perf_counter() - t_fetch0) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    # Output
    print(f"\nQuery: {args.q}")
    print(f"Results: {len(rows)} chunks\n")
    for (chunk_id, ticker, filing_date, item, txt, s0, s1, section_id, doc_id) in rows:
        snippet = re.sub(r"\s+", " ", txt)[: args.max_snippet_chars]
        cite = f"[{ticker} {str(filing_date)[:10]} • {item} • chunk {chunk_id} • doc {doc_id}]"
        print(f"- {cite} {snippet}…\n")

    # Timings
    print("Timings (ms): "
          f"model_load={t_model:.1f}, bm25={t_bm25:.1f}, ann={t_ann:.1f}, "
          f"fuse={t_fuse:.1f}, mmr={t_mmr:.1f}, fetch={t_fetch:.1f}, total={total_ms:.1f}")
    if not use_precomputed:
        print("(Note) Precomputed vectors not found; fell back to re-embedding candidates. "
              "Run export_chunk_vectors.py to generate chunk_ids.npy and chunk_embs.npy for much faster queries.")


if __name__ == "__main__":
    main()

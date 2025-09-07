#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sqlite3, duckdb, hnswlib, numpy as np, os, re, time
from sentence_transformers import SentenceTransformer

def rrf(a, b, k=60, C=60):
    score = {}
    for r, cid in enumerate(a): score[cid] = score.get(cid,0)+1.0/(C+r+1)
    for r, cid in enumerate(b): score[cid] = score.get(cid,0)+1.0/(C+r+1)
    return [cid for cid,_ in sorted(score.items(), key=lambda x:-x[1])][:k]

def mmr(qv, cand_vecs, k=8, lam=0.5):
    if len(cand_vecs)==0: return []
    sims = cand_vecs @ qv
    chosen, chosen_idx, avail = [], [], list(range(len(cand_vecs)))
    first = int(np.argmax(sims)); chosen.append(cand_vecs[first]); chosen_idx.append(first); avail.remove(first)
    while avail and len(chosen_idx) < k:
        scores = []
        for a in avail:
            redundancy = max(float(cand_vecs[a] @ c) for c in chosen)
            scores.append(lam*float(sims[a]) - (1-lam)*redundancy)
        pick = avail[int(np.argmax(scores))]
        chosen.append(cand_vecs[pick]); chosen_idx.append(pick); avail.remove(pick)
    return chosen_idx

class WarmRetriever:
    def __init__(self, duckdb_path, fts_path, hnsw_path, ids_npy, embs_npy, model_name,
                 ef=96, k_lex=80, k_ann=80, fuse_k=48, k_final=8, mmr_lambda=0.5):
        t0=time.perf_counter()
        self.duck = duckdb.connect(duckdb_path, read_only=True)
        self.sq   = sqlite3.connect(fts_path, check_same_thread=False)
        self.encoder = SentenceTransformer(model_name)
        # Precomputed vectors
        self.use_pre = os.path.exists(ids_npy) and os.path.exists(embs_npy)
        if self.use_pre:
            self.ids = np.load(ids_npy, mmap_mode="r")
            self.vecs = np.load(embs_npy, mmap_mode="r")
            dim = int(self.vecs.shape[1])
        else:
            dim = self.encoder.get_sentence_embedding_dimension()
        # HNSW
        self.idx = hnswlib.Index(space="cosine", dim=dim)
        self.idx.load_index(hnsw_path); self.idx.set_ef(ef)
        if self.use_pre and self.idx.dim != self.vecs.shape[1]:
            raise SystemExit(f"HNSW dim {self.idx.dim} != precomputed {self.vecs.shape[1]}")
        # id→row
        self.id2i = {int(cid): i for i, cid in enumerate(getattr(self, 'ids', []))}
        # Params
        self.k_lex, self.k_ann, self.fuse_k, self.k_final, self.mmr_lambda = k_lex, k_ann, fuse_k, k_final, mmr_lambda
        self.startup_ms = (time.perf_counter()-t0)*1000

    def query(self, q):
        t0=time.perf_counter()
        # BM25
        tb=time.perf_counter()
        lex_ids = [int(r[0]) for r in self.sq.execute(
            "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH ? LIMIT ?",(q,self.k_lex)
        ).fetchall()]
        bm_ms=(time.perf_counter()-tb)*1000
        # ANN
        ta=time.perf_counter()
        qv = self.encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        labels,_ = self.idx.knn_query(qv, k=self.k_ann)
        ann_ids = [int(x) for x in labels[0].tolist()]
        ann_ms=(time.perf_counter()-ta)*1000
        # RRF
        tf=time.perf_counter()
        cand_ids = rrf(lex_ids, ann_ids, k=max(self.fuse_k, self.k_final*4))
        fuse_ms=(time.perf_counter()-tf)*1000
        # MMR
        tm=time.perf_counter()
        if self.use_pre:
            cand_ids = [cid for cid in cand_ids if cid in self.id2i]
            cand_vecs = np.stack([self.vecs[self.id2i[cid]] for cid in cand_ids], axis=0) if cand_ids else np.empty((0,self.idx.dim))
        else:
            # fallback: re-embed candidate texts
            if cand_ids:
                rows = self.duck.execute(
                    f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({','.join(map(str,cand_ids))})"
                ).fetchall()
                id2txt = {int(cid):txt for cid,txt in rows}
                ordered = [id2txt[cid] for cid in cand_ids if cid in id2txt]
                cand_vecs = self.encoder.encode(ordered, normalize_embeddings=True, convert_to_numpy=True)
            else:
                cand_vecs = np.empty((0,self.idx.dim))
        idxs = mmr(qv, cand_vecs, k=self.k_final, lam=self.mmr_lambda)
        final_ids = [cand_ids[i] for i in idxs] if idxs else []
        mmr_ms=(time.perf_counter()-tm)*1000
        # Fetch metadata
        tg=time.perf_counter()
        rows=[]
        if final_ids:
            rows = self.duck.execute(f"""
            SELECT c.chunk_id, d.ticker, CAST(d.filing_date AS VARCHAR) AS filing_date,
                   s.item, c.text, c.start_sent_idx, c.end_sent_idx, c.section_id, d.doc_id
            FROM chunks c
            JOIN sections s ON c.section_id = s.section_id
            JOIN docs d     ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({','.join(map(str,final_ids))})
            """).fetchall()
        fetch_ms=(time.perf_counter()-tg)*1000
        total_ms=(time.perf_counter()-t0)*1000
        return rows, dict(bm25_ms=bm_ms, ann_ms=ann_ms, fuse_ms=fuse_ms, mmr_ms=mmr_ms, fetch_ms=fetch_ms, total_ms=total_ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="tenk.duckdb")
    ap.add_argument("--fts",    default="tenk_fts.sqlite")
    ap.add_argument("--hnsw",   default="tenk_hnsw.bin")
    ap.add_argument("--ids",    default="chunk_ids.npy")
    ap.add_argument("--embs",   default="chunk_embs.npy")
    ap.add_argument("--model",  default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k_lex", type=int, default=80)
    ap.add_argument("--k_ann", type=int, default=80)
    ap.add_argument("--k_final", type=int, default=8)
    ap.add_argument("--fuse_k", type=int, default=48)
    ap.add_argument("--mmr_lambda", type=float, default=0.5)
    args = ap.parse_args()

    retr = WarmRetriever(args.duckdb, args.fts, args.hnsw, args.ids, args.embs, args.model,
                         k_lex=args.k_lex, k_ann=args.k_ann, fuse_k=args.fuse_k, k_final=args.k_final, mmr_lambda=args.mmr_lambda)
    print(f"(startup) models+indexes loaded in {retr.startup_ms:.1f} ms")
    print("Type a query (or just Enter to exit):\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not q:
            print("bye!")
            break
        rows, t = retr.query(q)
        print(f"\nResults: {len(rows)} chunks")
        for (chunk_id,ticker,filing_date,item,txt,s0,s1,section_id,doc_id) in rows:
            snippet = re.sub(r"\s+"," ",txt)[:320]
            cite = f"[{ticker} {str(filing_date)[:10]} • {item} • chunk {chunk_id} • doc {doc_id}]"
            print(f"- {cite} {snippet}…")
        print(f"Timings (ms): bm25={t['bm25_ms']:.1f} ann={t['ann_ms']:.1f} fuse={t['fuse_ms']:.1f} "
              f"mmr={t['mmr_ms']:.1f} fetch={t['fetch_ms']:.1f} total={t['total_ms']:.1f}\n")

if __name__ == "__main__":
    main()

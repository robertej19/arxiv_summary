# warm_retriever.py
from __future__ import annotations
import sqlite3, duckdb, hnswlib, numpy as np, time, unicodedata
from sentence_transformers import SentenceTransformer

def _rrf(a: list[int], b: list[int], k: int = 60, C: int = 60) -> list[int]:
    s: dict[int, float] = {}
    for r, cid in enumerate(a): s[cid] = s.get(cid, 0.0) + 1.0/(C+r+1)
    for r, cid in enumerate(b): s[cid] = s.get(cid, 0.0) + 1.0/(C+r+1)
    return [cid for cid,_ in sorted(s.items(), key=lambda x: -x[1])][:k]

def _mmr(qv: np.ndarray, cand_vecs: np.ndarray, k: int = 8, lam: float = 0.5) -> list[int]:
    if cand_vecs.size == 0: return []
    sims = cand_vecs @ qv
    chosen, chosen_idx, avail = [], [], list(range(len(cand_vecs)))
    first = int(np.argmax(sims)); chosen.append(cand_vecs[first]); chosen_idx.append(first); avail.remove(first)
    while avail and len(chosen_idx) < k:
        sc = [lam*float(sims[a]) - (1-lam)*max(float(cand_vecs[a] @ c) for c in chosen) for a in avail]
        pick = avail[int(np.argmax(sc))]; chosen.append(cand_vecs[pick]); chosen_idx.append(pick); avail.remove(pick)
    return chosen_idx

def _normalize_fts_query(q: str) -> str:
    q = unicodedata.normalize("NFKC", q)
    q = (q.replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("‘", "'")
           .replace("–", "-").replace("—", "-"))
    return q.strip()

class WarmRetriever:
    def __init__(
        self,
        duckdb_path: str = "tenk.duckdb",
        fts_path: str = "tenk_fts.sqlite",
        hnsw_path: str = "tenk_hnsw.bin",
        ids_npy: str = "chunk_ids.npy",
        embs_npy: str = "chunk_embs.npy",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hnsw_ef: int = 96,
        k_lex: int = 80, k_ann: int = 80, fuse_k: int = 48, k_final: int = 8, mmr_lambda: float = 0.5,
    ):
        self.duck = duckdb.connect(duckdb_path, read_only=True)
        self.sq = sqlite3.connect(fts_path, check_same_thread=False)

        self.ids = np.load(ids_npy, mmap_mode="r")
        self.vecs = np.load(embs_npy, mmap_mode="r")
        self.id2i = {int(cid): i for i, cid in enumerate(self.ids)}
        dim = int(self.vecs.shape[1])

        self.idx = hnswlib.Index(space="cosine", dim=dim)
        self.idx.load_index(hnsw_path); self.idx.set_ef(hnsw_ef)
        if self.idx.dim != dim: raise ValueError("HNSW dim != embedding dim")

        self.encoder = SentenceTransformer(encoder_name)
        self.k_lex, self.k_ann, self.fuse_k, self.k_final, self.mmr_lambda = k_lex, k_ann, fuse_k, k_final, mmr_lambda

    def _bm25_ids(self, query: str, k: int) -> list[int]:
        q = _normalize_fts_query(query)
        q_sql = q.replace("'", "''")
        sql = f"SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH '{q_sql}' LIMIT ?"
        rows = self.sq.execute(sql, (k,)).fetchall()
        return [int(r[0]) for r in rows]

    def retrieve(self, query: str) -> tuple[list[tuple], dict]:
        t0=time.perf_counter()

        # BM25 (FTS5)
        tb=time.perf_counter()
        lex_ids = self._bm25_ids(query, self.k_lex)
        bm_ms=(time.perf_counter()-tb)*1000

        # ANN (HNSW) — get top similarity too (cosine = 1 - distance)
        ta=time.perf_counter()
        qv = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        labels, dists = self.idx.knn_query(qv, k=self.k_ann)
        ann_ids = [int(x) for x in labels[0].tolist()]
        ann_top_sim = float(1.0 - dists[0][0]) if len(ann_ids) else 0.0
        ann_ms=(time.perf_counter()-ta)*1000

        # Fusion (RRF)
        tf=time.perf_counter()
        cand_ids = _rrf(lex_ids, ann_ids, k=max(self.fuse_k, self.k_final*4))
        cand_ids = [cid for cid in cand_ids if cid in self.id2i]
        fuse_ms=(time.perf_counter()-tf)*1000

        # MMR (no re-embedding; use precomputed vectors)
        tm=time.perf_counter()
        cand_vecs = np.stack([self.vecs[self.id2i[cid]] for cid in cand_ids], axis=0) if cand_ids else np.empty((0,self.idx.dim))
        picks = _mmr(qv, cand_vecs, k=self.k_final, lam=self.mmr_lambda)
        final_ids = [cand_ids[i] for i in picks]
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
            JOIN docs d     ON c.doc_id     = d.doc_id
            WHERE c.chunk_id IN ({','.join(map(str,final_ids))})
            """).fetchall()
        fetch_ms=(time.perf_counter()-tg)*1000

        total_ms=(time.perf_counter()-t0)*1000
        return rows, {
            "bm25": bm_ms, "ann": ann_ms, "fuse": fuse_ms, "mmr": mmr_ms, "fetch": fetch_ms, "total": total_ms,
            "ann_top_sim": round(ann_top_sim, 4), "lex_count": len(lex_ids)
        }

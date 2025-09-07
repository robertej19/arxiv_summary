#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a dense HNSW index over 10-K chunks stored in DuckDB, and save minimal
chunk metadata for fast joins later.

Usage:
  python build_hnsw.py \
    --duckdb tenk.duckdb \
    --hnsw tenk_hnsw.bin \
    --meta chunk_meta.parquet \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch 512

Notes:
- Expects DuckDB tables: docs, sections, chunks (from the ingestion step).
- Embeddings are normalized for cosine similarity.
"""

import os
import argparse

import duckdb
import numpy as np
import hnswlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="tenk.duckdb", help="Path to DuckDB database")
    ap.add_argument("--hnsw",   default="tenk_hnsw.bin", help="Output HNSW index file")
    ap.add_argument("--meta",   default="chunk_meta.parquet", help="Output Parquet with chunk metadata")
    ap.add_argument("--model",  default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--batch",  type=int, default=512, help="Embedding batch size")
    ap.add_argument("--limit",  type=int, default=0, help="Optional limit during testing")
    ap.add_argument("--efc",    type=int, default=300, help="HNSW ef_construction")
    ap.add_argument("--M",      type=int, default=32, help="HNSW M (graph connections)")
    ap.add_argument("--ef",     type=int, default=96, help="HNSW ef (search)")
    args = ap.parse_args()

    # Ensure output folders exist
    for path in [args.hnsw, args.meta]:
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)

    # 1) Load rows to embed (chunk_id, text)
    con = duckdb.connect(args.duckdb)
    q = "SELECT chunk_id, text FROM chunks"
    if args.limit > 0:
        q += f" LIMIT {args.limit}"
    rows = con.execute(q).fetchall()
    n = len(rows)
    if n == 0:
        raise SystemExit("No rows found in chunks table. Did you run ingestion?")

    print(f"Embedding {n} chunks with model: {args.model}")

    # 2) Model & HNSW init
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.efc, M=args.M)

    ids = np.empty(n, dtype=np.int64)
    embs = np.empty((n, dim), dtype=np.float32)

    # 3) Encode in batches (normalized embeddings)
    for i in tqdm(range(0, n, args.batch), desc="Embedding"):
        batch = rows[i : i + args.batch]
        ids[i : i + len(batch)] = [int(r[0]) for r in batch]
        texts = [r[1] for r in batch]
        E = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        embs[i : i + len(batch)] = E

    # 4) Build index
    index.add_items(embs, ids)
    index.set_ef(args.ef)
    index.save_index(args.hnsw)

    # 5) Persist minimal metadata (DuckDB COPY can't use parameter placeholders)
    meta_path_sql = args.meta.replace("'", "''")
    con.execute(f"""
    COPY (
      SELECT
        c.chunk_id,
        c.doc_id,
        c.section_id,
        s.item       AS section,
        d.filing_date AS filing_date
      FROM chunks c
      JOIN sections s ON c.section_id = s.section_id
      JOIN docs d     ON c.doc_id     = d.doc_id
    ) TO '{meta_path_sql}' (FORMAT PARQUET)
    """)

    print(f"HNSW saved → {args.hnsw}")
    print(f"Chunk meta → {args.meta}")
    print("Done.")


if __name__ == "__main__":
    main()

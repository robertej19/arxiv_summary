# export_chunk_vectors.py
#!/usr/bin/env python3
import argparse, duckdb, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--duckdb", default="tenk.duckdb")
ap.add_argument("--model",  default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--batch",  type=int, default=512)
ap.add_argument("--ids",    default="chunk_ids.npy")
ap.add_argument("--embs",   default="chunk_embs.npy")
ap.add_argument("--limit",  type=int, default=0)
args = ap.parse_args()

con = duckdb.connect(args.duckdb)
q = "SELECT chunk_id, text FROM chunks"
if args.limit > 0: q += f" LIMIT {args.limit}"
rows = con.execute(q).fetchall()
if not rows: raise SystemExit("No chunks found.")

model = SentenceTransformer(args.model)
dim = model.get_sentence_embedding_dimension()
ids  = np.empty(len(rows), dtype=np.int64)
embs = np.empty((len(rows), dim), dtype=np.float32)

for i in tqdm(range(0, len(rows), args.batch), desc="Embedding"):
    batch = rows[i:i+args.batch]
    ids[i:i+len(batch)] = [int(r[0]) for r in batch]
    texts = [r[1] for r in batch]
    E = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embs[i:i+len(batch)] = E

np.save(args.ids,  ids)
np.save(args.embs, embs.astype(np.float32))
print(f"Wrote {args.ids} and {args.embs}")

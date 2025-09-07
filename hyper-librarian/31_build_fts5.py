#!/usr/bin/env python3
import sqlite3, duckdb, argparse, os, sys
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--duckdb", default="tenk.duckdb")
ap.add_argument("--fts", default="tenk_fts.sqlite")
ap.add_argument("--limit", type=int, default=0, help="Optional limit for speed during testing")
args = ap.parse_args()

if os.path.exists(args.fts): os.remove(args.fts)

con = duckdb.connect(args.duckdb)
sq  = sqlite3.connect(args.fts)
sq.execute("PRAGMA journal_mode=WAL;")
sq.execute("PRAGMA synchronous=OFF;")
sq.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(chunk_id UNINDEXED, text, section UNINDEXED, year UNINDEXED, tokenize='porter');")

q = """
SELECT c.chunk_id, c.text, s.item as section, CAST(d.filing_date AS VARCHAR) as year
FROM chunks c
JOIN sections s ON c.section_id = s.section_id
JOIN docs d     ON c.doc_id     = d.doc_id
"""
if args.limit > 0:
    q += f" LIMIT {args.limit}"

rows = con.execute(q).fetchall()
for chunk_id, text, section, year in tqdm(rows, desc="FTS insert"):
    sq.execute("INSERT INTO fts_chunks(chunk_id,text,section,year) VALUES (?,?,?,?)",
               (int(chunk_id), text, section, (year or "")[:4]))
sq.commit()
sq.close()
print(f"FTS5 built → {args.fts}")

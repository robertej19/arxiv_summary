# sanity_check_paths.py
import os, json, glob
IN = "../corpus_arxiv_rl"  # adjust if needed
missing = 0; total = 0
for mp in glob.glob(f"{IN}/meta/*.json"):
    m = json.load(open(mp))
    if not os.path.exists(m["pdf_path"]):
        missing += 1
    total += 1
print(f"meta files: {total}, missing pdf paths: {missing}")

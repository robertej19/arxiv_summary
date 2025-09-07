# local_corpus/ingest.py
import os, json, re, glob, argparse
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

CHUNK_MAX = 1600
CHUNK_OVERLAP = 240

def norm_whitespace(s: str) -> str:
    s = re.sub(r'\u00AD', '', s)            # soft hyphen
    s = re.sub(r'-\n', '', s)               # hyphen line-break
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def chunk_text(text: str, max_len=CHUNK_MAX, overlap=CHUNK_OVERLAP):
    paras = re.split(r'\n{2,}', text)
    acc, chunks = "", []
    for p in paras:
        if len(acc) + len(p) + 2 <= max_len:
            acc = (acc + "\n\n" + p).strip() if acc else p
        else:
            if acc: chunks.append(acc)
            if len(p) > max_len:
                sents = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z(])', p)
                cur = ""
                for s in sents:
                    if len(cur) + len(s) + 1 <= max_len:
                        cur = (cur + " " + s).strip() if cur else s
                    else:
                        if cur: chunks.append(cur)
                        cur = s
                if cur: chunks.append(cur)
            else:
                chunks.append(p)
            acc = ""
    if acc: chunks.append(acc)

    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            prev_tail = out[-1][-overlap:]
            merged = (prev_tail + " " + c).strip()
            out.append(merged[:max_len])
    return out

def extract_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        pages.append(norm_whitespace(txt))
    return pages

def resolve_pdf_path(in_dir: str, meta: dict) -> str | None:
    """Prefer current-tree path; fall back to stored path; last resort: glob by base id."""
    arxid = meta["arxiv_id"]  # e.g., 2101.01234v3
    base = arxid.split("v")[0]
    # 1) expected current location
    guess = os.path.join(in_dir, "pdf", f"{arxid}.pdf")
    if os.path.exists(guess):
        return guess
    # 2) stored path from metadata
    stored = meta.get("pdf_path")
    if stored and os.path.exists(stored):
        return stored
    # 3) any version under this base id in current tree
    candidates = sorted(glob.glob(os.path.join(in_dir, "pdf", f"{base}v*.pdf")))
    if candidates:
        return candidates[-1]  # pick highest v* lexicographically
    return None

def run(in_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta_files = sorted(glob.glob(os.path.join(in_dir, "meta", "*.json")))
    out_path = os.path.join(out_dir, "chunks.jsonl")
    map_path = os.path.join(out_dir, "chunk_map.json")

    n_chunks = 0
    mapping = []
    c_total = len(meta_files)
    c_missing_pdf = 0
    c_read_fail = 0
    c_empty_text = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for mp in tqdm(meta_files, desc="Ingest"):
            try:
                m = json.load(open(mp, "r", encoding="utf-8"))
            except Exception:
                continue

            pdf = resolve_pdf_path(in_dir, m)
            if not pdf:
                c_missing_pdf += 1
                continue

            try:
                pages = extract_pdf(pdf)
            except Exception:
                c_read_fail += 1
                continue

            fulltext = "\n\n".join(pages).strip()
            if not fulltext:
                c_empty_text += 1
                continue

            chunks = chunk_text(fulltext, CHUNK_MAX, CHUNK_OVERLAP)
            for idx, c in enumerate(chunks):
                rec = {
                    "doc_id": m["arxiv_id"],
                    "title": m.get("title"),
                    "year": int(m["published"][:4]) if m.get("published") else None,
                    "chunk_id": f"{m['arxiv_id']}::c{idx}",
                    "text": c,
                    "source": "arxiv",
                    "pdf_path": pdf,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mapping.append({
                    "chunk_id": rec["chunk_id"],
                    "doc_id": rec["doc_id"],
                    "title": rec["title"],
                    "year": rec["year"],
                    "pdf_path": pdf,
                })
                n_chunks += 1

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[ingest] meta files: {c_total}")
    print(f"[ingest] missing PDFs: {c_missing_pdf}")
    print(f"[ingest] read failures: {c_read_fail}")
    print(f"[ingest] empty-text PDFs: {c_empty_text}")
    print(f"[ingest] wrote {n_chunks} chunks to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="../corpus_arxiv_rl", help="Directory with pdf/ and meta/")
    ap.add_argument("--out-dir", default="./store", help="Output store dir")
    args = ap.parse_args()
    run(args.in_dir, args.out_dir)

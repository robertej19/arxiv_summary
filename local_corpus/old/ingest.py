# local_corpus/ingest.py
import os, json, re, glob
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

IN_DIR = "../corpus_arxiv_rl"        # has pdf/ and meta/
OUT_DIR = "./store"
CHUNK_MAX = 1600                      # ~chars per chunk
CHUNK_OVERLAP = 240                   # ~chars overlap

os.makedirs(OUT_DIR, exist_ok=True)

def norm_whitespace(s: str) -> str:
    s = re.sub(r'\u00AD', '', s)            # soft hyphen
    s = re.sub(r'-\n', '', s)               # hyphen line-break
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def chunk_text(text: str, max_len=CHUNK_MAX, overlap=CHUNK_OVERLAP):
    # simple char-based chunker, tries to cut on paragraph or sentence ends
    paras = re.split(r'\n{2,}', text)
    acc, chunks = "", []
    for p in paras:
        if len(acc) + len(p) + 2 <= max_len:
            acc = (acc + "\n\n" + p).strip() if acc else p
        else:
            # flush acc smartly
            if acc:
                chunks.append(acc)
            # if paragraph itself is too long, split by sentence-ish
            if len(p) > max_len:
                sents = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z(])', p)
                cur = ""
                for s in sents:
                    if len(cur) + len(s) + 1 <= max_len:
                        cur = (cur + " " + s).strip() if cur else s
                    else:
                        if cur:
                            chunks.append(cur)
                        cur = s
                if cur:
                    chunks.append(cur)
            else:
                chunks.append(p)
            acc = ""
    if acc:
        chunks.append(acc)

    # add overlap
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            prev_tail = out[-1][-overlap:]
            merged = (prev_tail + " " + c).strip()
            # keep size in check
            out[-1] = out[-1]  # previous remains same
            out.append(merged[:max_len])
    return out

def extract_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        pages.append(norm_whitespace(txt))
    return pages

def run():
    meta_files = sorted(glob.glob(os.path.join(IN_DIR, "meta", "*.json")))
    out_path = os.path.join(OUT_DIR, "chunks.jsonl")
    map_path = os.path.join(OUT_DIR, "chunk_map.json")
    n_chunks = 0
    mapping = []  # maps row_index -> (arxiv_id, title, pagespan)

    with open(out_path, "w", encoding="utf-8") as out:
        for mp in tqdm(meta_files, desc="Ingest"):
            with open(mp, "r", encoding="utf-8") as f:
                m = json.load(f)

            pdf = m["pdf_path"]
            if not os.path.exists(pdf):
                continue

            try:
                pages = extract_pdf(pdf)
            except Exception as e:
                print(f"[warn] failed to read {pdf}: {e}")
                continue

            fulltext = "\n\n".join(pages)
            if not fulltext.strip():
                continue

            chunks = chunk_text(fulltext, CHUNK_MAX, CHUNK_OVERLAP)
            for idx, c in enumerate(chunks):
                rec = {
                    "doc_id": m["arxiv_id"],
                    "title": m["title"],
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

    print(f"[ingest] wrote {n_chunks} chunks to {out_path}")

if __name__ == "__main__":
    run()

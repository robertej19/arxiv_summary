#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, hashlib, argparse
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm
import duckdb
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Ensure both resources are available
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

nltk.download('punkt', quiet=True)

# ---------- Helpers ----------
ITEM_RE = re.compile(
    r'(?im)^\s*item\s*[\-:\.]?\s*((?:1a|1b|1|7a|7|8|9a|9|10|11|12|13|14))\s*[\-:\.]?\s*(.*)$'
)
CANON = {
    '1': 'Business',
    '1a': 'Risk Factors',
    '1b': 'Unresolved Staff Comments',
    '7': 'MD&A',
    '7a': 'Market Risk',
    '8': 'Financial Statements',
    '9a': 'Controls and Procedures',
    '9': 'Changes in Disagreements',
    '10': 'Directors/Execs',
    '11': 'Executive Compensation',
    '12': 'Security Ownership',
    '13': 'Certain Relationships',
    '14': 'Principal Accountant Fees'
}

def sha1(s:str)->str: return hashlib.sha1(s.encode('utf-8')).hexdigest()

def html_to_text(html: str) -> str:
    # Strip scripts/styles, keep text with spacing
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript"]): tag.extract()
    # Replace <br> with newlines to preserve basic structure
    for br in soup.find_all("br"): br.replace_with("\n")
    text = soup.get_text("\n")
    # collapse excessive whitespace
    text = re.sub(r'\u00a0', ' ', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def find_sections(txt: str) -> List[Tuple[str,int,int]]:
    hits = [(m.start(), m.group(1).lower()) for m in ITEM_RE.finditer(txt)]
    hits.sort(key=lambda x: x[0])
    sections = []
    for i, (start, code) in enumerate(hits):
        end = hits[i+1][0] if i+1 < len(hits) else len(txt)
        sections.append((CANON.get(code, code.upper()), start, end))
    # ensure at least one big catch-all if nothing matched
    if not sections:
        sections = [("FULL_DOCUMENT", 0, len(txt))]
    return sections

def sentence_spans(text: str) -> List[Tuple[int,int,str]]:
    spans = []
    pos = 0
    for s in sent_tokenize(text):
        s = s.strip()
        idx = text.find(s, pos)
        if idx == -1:  # fallback search
            idx = text.find(s)
            if idx == -1: continue
        start = idx; end = idx + len(s)
        spans.append((start, end, s))
        pos = end
    return spans

def make_chunks(section_text: str, target_tokens=420, overlap_sents=2):
    sents = []
    pos = 0
    for s in sent_tokenize(section_text):
        s = s.strip()
        idx = section_text.find(s, pos)
        if idx == -1: idx = section_text.find(s)
        st = max(0, idx); en = st + len(s); pos = en
        sents.append((s, st, en))
    chunks = []
    cur, cur_tok = [], 0
    for i,(s,st,en) in enumerate(sents):
        t = max(1,len(s.split()))
        if cur and cur_tok + t > target_tokens:
            chunks.append(cur)
            cur = cur[-overlap_sents:] if overlap_sents>0 else []
            cur_tok = sum(len(x[0].split()) for x in cur)
        cur.append((s,st,en,i))
        cur_tok += t
    if cur: chunks.append(cur)
    return sents, chunks

def parse_filename(p: Path):
    # e.g. 10k_filings/GOOGL/2024-01-31_GOOGL_10-K_goog-20231231.htm
    # Extract ticker from folder or filename, and filing_date from prefix
    ticker = p.parent.name.upper()
    m = re.match(r'(\d{4}-\d{2}-\d{2})_([A-Za-z\.\-]+)_10-K', p.name)
    filing_date = m.group(1) if m else None
    return ticker, filing_date

# ---------- Main pipeline ----------
def ingest(root_dir: str, duck_path: str = "tenk.duckdb"):
    con = duckdb.connect(duck_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS docs(
      doc_id BIGINT PRIMARY KEY,
      path TEXT,
      ticker TEXT,
      filing_date DATE,
      name TEXT,
      text_len BIGINT
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS sections(
      section_id BIGINT PRIMARY KEY,
      doc_id BIGINT,
      item TEXT,
      start_char BIGINT,
      end_char BIGINT
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS sentences(
      sent_id BIGINT PRIMARY KEY,
      doc_id BIGINT,
      section_id BIGINT,
      sent_idx INT,
      start_char BIGINT,
      end_char BIGINT,
      text TEXT,
      sent_hash TEXT
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
      chunk_id BIGINT PRIMARY KEY,
      doc_id BIGINT,
      section_id BIGINT,
      start_sent_idx INT,
      end_sent_idx INT,
      text TEXT,
      token_count INT
    );
    """)

    doc_id = con.execute("SELECT COALESCE(MAX(doc_id),0) FROM docs").fetchone()[0] or 0
    section_id = con.execute("SELECT COALESCE(MAX(section_id),0) FROM sections").fetchone()[0] or 0
    sent_id = con.execute("SELECT COALESCE(MAX(sent_id),0) FROM sentences").fetchone()[0] or 0
    chunk_id = con.execute("SELECT COALESCE(MAX(chunk_id),0) FROM chunks").fetchone()[0] or 0

    files = [p for p in Path(root_dir).rglob("*.htm*")]
    for p in tqdm(files, desc="Ingesting 10-K HTML"):
        html = p.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        if not text or len(text) < 1000:
            continue

        ticker, filing_date = parse_filename(p)
        doc_id += 1
        con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)",
                    (doc_id, str(p), ticker, filing_date, p.name, len(text)))

        # Sections
        secs = find_sections(text)
        sec_ids = []
        for item, st, en in secs:
            section_id += 1
            sec_ids.append((section_id, st, en, item))
            con.execute("INSERT INTO sections VALUES (?,?,?,?,?)",
                        (section_id, doc_id, item, st, en))

        # Sentences + chunks (per section)
        for sid, st, en, item in sec_ids:
            sec_txt = text[st:en]
            sents, chunks = make_chunks(sec_txt, target_tokens=420, overlap_sents=2)
            # sentences
            for j,(s, sst, sen) in enumerate(sents):
                sent_id += 1
                con.execute("INSERT INTO sentences VALUES (?,?,?,?,?,?,?,?)",
                            (sent_id, doc_id, sid, j, st+sst, st+sen, s, sha1(s)))
            # chunks
            for ch in chunks:
                start_idx = ch[0][3]; end_idx = ch[-1][3]
                chunk_text = " ".join(x[0] for x in ch)
                tok_ct = sum(len(x[0].split()) for x in ch)
                chunk_id += 1
                con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                            (chunk_id, doc_id, sid, start_idx, end_idx, chunk_text, tok_ct))
    con.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="10k_filings", help="Root folder of 10-K HTML files")
    ap.add_argument("--duckdb", type=str, default="tenk.duckdb", help="DuckDB output path")
    args = ap.parse_args()
    ingest(args.root, args.duckdb)
    print("Ingest complete → tables: docs, sections, sentences, chunks")

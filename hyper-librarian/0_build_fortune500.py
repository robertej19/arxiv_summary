#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate companies.txt (CIK,TICKER,COMPANY_NAME) by resolving a provided list of company
names against the SEC's official company_tickers.json.

Usage examples:

  # Preferred: provide your Fortune 500 CSV with a 'Company' (or 'Name') column
  python build_fortune500.py --names-csv fortune500_2025.csv --out companies.txt

  # Quick pipeline test (uses S&P 500 from Wikipedia):
  python build_fortune500.py --use-sp500 --out companies.txt

  # Accept borderline matches automatically (merge review.csv into companies.txt):
  python build_fortune500.py --use-sp500 --out companies.txt --accept-review

Notes:
- Private companies (no SEC filings) will be written to skipped.csv.
- Borderline fuzzy matches go to review.csv unless --accept-review is provided.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import requests
from rapidfuzz import process, fuzz

# ------------------------ Config / Defaults ------------------------

BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SEC_UA = {"User-Agent": "HyperLibrarian/1.0 (contact: team@example.com)"}

FILE_OUT_DEFAULT        = "companies.txt"
FILE_REVIEW_DEFAULT     = "review.csv"
FILE_SKIPPED_DEFAULT    = "skipped.csv"

THRESH_STRICT_DEFAULT   = 92   # auto-accept >= this score
THRESH_REVIEW_DEFAULT   = 85   # send to review if score in [REVIEW, STRICT)
MAX_CANDIDATES          = 5

# ------------------------ Data structures ------------------------

@dataclass
class SecRow:
    cik: str         # zero-padded 10 digits
    ticker: str      # upper-case ticker
    name: str        # EDGAR conformed name
    norm: str        # normalized for fuzzy matching

# ------------------------ Helpers ------------------------

def normalize_name(s: str) -> str:
    """Lowercase, strip corporate suffixes/symbols to improve fuzzy matching."""
    s0 = s.lower()
    s0 = re.sub(r"&", " and ", s0)
    s0 = re.sub(r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|plc|holdings|group|the)\b\.?", "", s0)
    s0 = re.sub(r"[^a-z0-9\s\-\.\&]", "", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0

def fetch_sec_company_tickers() -> List[SecRow]:
    """Download SEC company_tickers.json and return normalized rows."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_UA, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows: List[SecRow] = []
    for _, rec in data.items():
        cik = f"{int(rec.get('cik_str')):010d}"
        ticker = (rec.get("ticker") or "").upper()
        name = rec.get("title") or ""
        rows.append(SecRow(cik=cik, ticker=ticker, name=name, norm=normalize_name(name)))
    return rows

def best_match(
    name: str,
    sec_rows: List[SecRow],
    review_threshold: int,
    limit: int = MAX_CANDIDATES,
) -> Tuple[Optional[SecRow], int, List[Tuple[int, SecRow]]]:
    """
    Fuzzy-match a company name to SEC rows.
    Returns (best_row_or_None_if_below_review_threshold, score, [(score, SecRow), ...]).
    """
    q = normalize_name(name)
    choices = [sr.norm for sr in sec_rows]
    results = process.extract(q, choices, scorer=fuzz.WRatio, limit=limit)
    candidates: List[Tuple[int, SecRow]] = []
    best: Optional[SecRow] = None
    best_score = -1
    for _, score, idx in results:
        sr = sec_rows[idx]
        candidates.append((score, sr))
        if score > best_score:
            best_score = score
            best = sr
    return (best if best_score >= review_threshold else None), best_score, candidates

def write_companies_txt(path: str, rows: List[Tuple[str, str, str]]) -> None:
    """Write header + CIK,TICKER,NAME rows."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# Format: CIK,TICKER,COMPANY_NAME\n")
        for cik, tk, nm in rows:
            f.write(f"{cik},{tk},{nm}\n")

def write_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

def load_company_names_from_csv(path: str) -> List[str]:
    """Load seed names from a CSV with Company/Name column variants."""
    df = pd.read_csv(path)
    for col in ["Company", "company", "Name", "name", "Company Name", "company_name"]:
        if col in df.columns:
            names = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
            if names:
                return names
    raise SystemExit("CSV must include a 'Company' (or Name) column with company names.")

def load_sp500_names_from_wikipedia() -> List[str]:
    """
    Robustly fetch S&P 500 constituent names from Wikipedia.
    Requires: requests, pandas, lxml
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    last_err = None
    for _ in range(3):
        try:
            resp = requests.get(url, headers=BROWSER_UA, timeout=30)
            resp.raise_for_status()
            html = resp.text
            tables = pd.read_html(io.StringIO(html))
            for t in tables:
                cols = {str(c).strip().lower() for c in t.columns}
                if "symbol" in cols and "security" in cols:
                    names = [str(x).strip() for x in t["Security"].tolist() if str(x).strip()]
                    if len(names) >= 400:
                        return names
            raise RuntimeError("S&P 500 table not found in parsed HTML.")
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Failed to fetch S&P 500 names: {last_err}")

def append_review_to_companies(companies_path: str, review_path: str) -> int:
    """
    Append rows from review.csv to companies.txt, robust to column-name variants.
    Returns number of rows appended.
    """
    if not os.path.exists(review_path):
        return 0
    df = pd.read_csv(review_path)
    if df.empty:
        return 0

    # Normalize headers (case-insensitive)
    norm = {c.strip().lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm:
                return norm[c]
        return None

    cik_col    = pick("cik", "cik_str", "sec_cik", "sec_cik_str")
    ticker_col = pick("ticker", "symbol")
    name_col   = pick("matchededgarname", "bestname", "edgar_name", "edgarname", "name", "title", "company", "conformedname")

    if not (cik_col and ticker_col and name_col):
        raise ValueError(f"review.csv missing needed columns; found: {list(df.columns)}")

    # Read existing triples to avoid dupes
    existing = set()
    if os.path.exists(companies_path):
        with open(companies_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                existing.add(tuple(line.strip().split(",", 2)))

    added = 0
    with open(companies_path, "a", encoding="utf-8") as out:
        for _, r in df.iterrows():
            cik = "".join(ch for ch in str(r[cik_col]) if ch.isdigit()).zfill(10)
            ticker = str(r[ticker_col]).upper().strip()
            name = str(r[name_col]).strip()
            triple = (cik, ticker, name)
            if triple in existing or not cik or not ticker or not name:
                continue
            out.write(f"{cik},{ticker},{name}\n")
            existing.add(triple)
            added += 1
    return added

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names-csv", type=str, help="CSV with a 'Company' (or 'Name') column listing Fortune 500 names")
    ap.add_argument("--use-sp500", action="store_true", help="Use S&P 500 (public) as a fallback source for testing")
    ap.add_argument("--out", type=str, default=FILE_OUT_DEFAULT, help="Output file (CIK,TICKER,COMPANY_NAME)")
    ap.add_argument("--review-out", type=str, default=FILE_REVIEW_DEFAULT, help="Borderline matches to review")
    ap.add_argument("--skipped-out", type=str, default=FILE_SKIPPED_DEFAULT, help="Names with no acceptable SEC match")
    ap.add_argument("--strict-threshold", type=int, default=THRESH_STRICT_DEFAULT, help="Auto-accept threshold (default 92)")
    ap.add_argument("--review-threshold", type=int, default=THRESH_REVIEW_DEFAULT, help="Review threshold (default 85)")
    ap.add_argument("--accept-review", action="store_true", help="Append all review.csv rows into companies.txt")
    args = ap.parse_args()

    # quick sanity on path args to avoid type collisions
    for p in (args.out, args.review_out, args.skipped_out):
        if not isinstance(p, str):
            raise SystemExit(f"Expected string path, got {type(p)} for {p}")

    if not args.names_csv and not args.use_sp500:
        print("Provide --names-csv fortune500.csv (preferred) or --use-sp500 (testing).", file=sys.stderr)
        sys.exit(2)

    # Load seed names
    if args.names_csv:
        seed_names = load_company_names_from_csv(args.names_csv)
    else:
        seed_names = load_sp500_names_from_wikipedia()

    # SEC mapping
    sec_rows = fetch_sec_company_tickers()

    # Match
    accepted: List[Tuple[str, str, str]] = []
    review_rows: List[List[str]] = []
    skipped_rows: List[List[str]] = []

    for company in seed_names:
        best, score, cands = best_match(
            company, sec_rows, review_threshold=args.review_threshold, limit=MAX_CANDIDATES
        )
        if best and score >= args.strict_threshold:
            accepted.append((best.cik, best.ticker, best.name))
        elif best and args.review_threshold <= score < args.strict_threshold:
            cand_str = " | ".join([f"{s}:{c.name} ({c.ticker}/{c.cik})" for s, c in cands])
            review_rows.append([company, best.name, str(score), best.cik, best.ticker, cand_str])
        else:
            skipped_rows.append([company])

    # Dedup + sort + write
    deduped = sorted(list({row for row in accepted}), key=lambda x: x[2].lower())
    write_companies_txt(args.out, deduped)

    if review_rows:
        write_csv(
            args.review_out,
            ["InputName", "MatchedEDGARName", "Score", "CIK", "Ticker", "Candidates"],
            review_rows,
        )
    if skipped_rows:
        write_csv(args.skipped_out, ["InputName"], skipped_rows)

    print(f"Wrote {len(deduped)} matches to {args.out}")
    if review_rows:
        print(f"Borderline matches to review: {len(review_rows)} → {args.review_out}")
    if skipped_rows:
        print(f"Unmatched names (likely private / non-SEC): {len(skipped_rows)} → {args.skipped_out}")

    if args.accept_review and review_rows:
        added = append_review_to_companies(args.out, args.review_out)
        print(f"Appended {added} review rows into {args.out}")

if __name__ == "__main__":
    main()

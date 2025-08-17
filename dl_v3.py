#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download a small offline arXiv corpus of Reinforcement Learning papers.
- Shards by year & category to avoid pagination hiccups
- Uses arxiv.Client.results (new API) with page_size/delay/retries
- Correct submittedDate format: [YYYYMMDDHHMM+TO+YYYYMMDDHHMM]
- Optional probe mode (count results, don't download)
- RL keyword filter (tweakable)
- Dedup to latest version per base_id (vN)
"""

import argparse
import json
import os
import re
import time
from typing import List

import arxiv
from tenacity import retry, wait_fixed, stop_after_attempt
from tqdm import tqdm


# ----------------------------
# Query construction
# ----------------------------

# High-precision RL keywords; you can loosen/tighten as needed
RL_KEYWORDS = [
    "reinforcement learning", "policy gradient", "actor-critic",
    "q-learning", "temporal difference", "ppo", "trpo", "ddpg",
    "sac", "td3", "a3c", "a2c", "dqn", "offline rl", "model-based",
    "world model", "intrinsic reward", "exploration bonus", "credit assignment",
]

BASE_Q = (
    '('
    'ti:"reinforcement learning" OR abs:"reinforcement learning" '
    'OR ti:"policy gradient" OR abs:"policy gradient" '
    'OR ti:PPO OR abs:PPO OR ti:TRPO OR abs:TRPO OR ti:DDPG OR abs:DDPG '
    'OR ti:SAC OR abs:SAC OR ti:TD3 OR abs:TD3 OR ti:DQN OR abs:DQN'
    ')'
)

def submitted_range(year: int) -> str:
    # Whole-year UTC bounds with required +TO+ separator
    start = f"{year}01010000"
    end   = f"{year}12312359"
    return f"submittedDate:[{start}+TO+{end}]"

def build_query(cat: str, year: int) -> str:
    # Keep category outside BASE_Q to avoid duplication
    return f"({BASE_Q}) AND cat:{cat} AND {submitted_range(year)}"


# ----------------------------
# Client & download helpers
# ----------------------------

def make_client(delay: int = 3, retries: int = 5, page_size: int = 50) -> arxiv.Client:
    return arxiv.Client(page_size=page_size, delay_seconds=delay, num_retries=retries)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def safe_download(result: arxiv.Result, path: str):
    result.download_pdf(filename=path)

def is_rl_paper(title: str, abstract: str, include_surveys: bool) -> bool:
    t = f"{title} || {abstract}".lower()
    if not include_surveys and ("survey" in t or "systematic review" in t):
        return False
    return any(k in t for k in RL_KEYWORDS)

def ensure_dirs(out_dir: str):
    os.makedirs(os.path.join(out_dir, "pdf"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

def dedupe_latest_versions(out_dir: str, verbose: bool = True):
    """Keep only latest vN per base_id; remove older version PDFs+JSON."""
    meta_dir = os.path.join(out_dir, "meta")
    pdf_dir = os.path.join(out_dir, "pdf")
    metas = [os.path.join(meta_dir, f) for f in os.listdir(meta_dir) if f.endswith(".json")]

    best = {}  # base_id -> (vnum, meta_path)
    for mp in metas:
        try:
            with open(mp) as f:
                m = json.load(f)
        except Exception:
            continue
        arxid = m.get("arxiv_id", "")
        if "v" not in arxid:
            continue
        base, v = arxid.split("v")
        vnum = int(v) if v.isdigit() else 1
        if base not in best or vnum > best[base][0]:
            best[base] = (vnum, mp)

    removed = 0
    for mp in metas:
        try:
            with open(mp) as f:
                m = json.load(f)
        except Exception:
            continue
        arxid = m.get("arxiv_id", "")
        if "v" not in arxid:
            continue
        base, _ = arxid.split("v")
        if mp != best.get(base, (None, None))[1]:
            # remove this older version
            pdf = m.get("pdf_path", "")
            for p in [mp, pdf]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        removed += 1
                    except FileNotFoundError:
                        pass
    if verbose:
        print(f"[dedupe] Removed {removed} older-version files.")

def probe_shard(client: arxiv.Client, category: str, year: int, max_results: int) -> int:
    q = build_query(category, year)
    search = arxiv.Search(
        query=q,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    n = 0
    for _ in client.results(search):
        n += 1
    return n


# ----------------------------
# Main download routine
# ----------------------------

def run(
    out_dir: str,
    years: List[int],
    categories: List[str],
    max_results_per_shard: int,
    include_surveys: bool,
    probe_only: bool,
    delay: int,
    retries: int,
    page_size: int,
):
    ensure_dirs(out_dir)
    client = make_client(delay=delay, retries=retries, page_size=page_size)

    total_kept = 0
    for year in years:
        for cat in categories:
            q = build_query(cat, year)
            search = arxiv.Search(
                query=q,
                max_results=max_results_per_shard,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            desc = f"{year}-{cat}"
            shard_kept = 0

            # In probe mode: just count and print
            if probe_only:
                try:
                    count = probe_shard(client, cat, year, max_results_per_shard)
                    print(f"{desc}: probe found {count} results")
                except arxiv.UnexpectedEmptyPageError as e:
                    print(f"[probe warn] Empty page for {desc}: {e}")
                except Exception as e:
                    print(f"[probe warn] Other error for {desc}: {e}")
                continue

            # Download loop with backoff on empty pages
            while True:
                try:
                    for r in tqdm(client.results(search), desc=desc, leave=False):
                        if not is_rl_paper(r.title, r.summary, include_surveys):
                            continue

                        arxid = r.get_short_id()            # e.g., 2101.01234v3
                        base_id = arxid.split("v")[0]
                        pdf_path = os.path.join(out_dir, "pdf", f"{arxid}.pdf")
                        meta_path = os.path.join(out_dir, "meta", f"{arxid}.json")

                        # Download
                        if not os.path.exists(pdf_path):
                            try:
                                safe_download(r, pdf_path)
                            except Exception as e:
                                print(f"[download warn] {arxid} failed: {e}")
                                continue

                        # Metadata
                        meta = {
                            "arxiv_id": arxid,
                            "base_id": base_id,
                            "title": r.title,
                            "abstract": r.summary,
                            "categories": list(r.categories),
                            "primary_category": r.primary_category,
                            "authors": [a.name for a in r.authors],
                            "published": r.published.strftime("%Y-%m-%d"),
                            "updated": r.updated.strftime("%Y-%m-%d"),
                            "pdf_path": pdf_path,
                            "links": [str(l.href) for l in r.links],
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        shard_kept += 1
                        total_kept += 1

                    break  # shard finished

                except arxiv.UnexpectedEmptyPageError as e:
                    # backoff and retry this shard
                    print(f"\n[warn] Empty page on {desc}. Backing off: {e}")
                    time.sleep(5)
                    client = make_client(delay=max(delay * 2, 6), retries=max(retries, 8), page_size=max(10, page_size // 2))

                except Exception as e:
                    print(f"\n[warn] Other error on {desc}: {e}")
                    time.sleep(3)
                    client = make_client(delay=max(delay * 2, 6), retries=max(retries, 8), page_size=max(10, page_size // 2))

            print(f"{desc}: kept {shard_kept}")

    # Final dedupe pass (only when actually downloading)
    if not probe_only:
        dedupe_latest_versions(out_dir, verbose=True)

    print(f"Done. Total kept (pre-dedupe count): {total_kept}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download a local arXiv RL corpus (PDF + JSON metadata).")
    p.add_argument("--out", default="corpus_arxiv_rl", help="Output directory (default: corpus_arxiv_rl)")
    p.add_argument("--years", default="2018-2022", help="Year or range, e.g. 2018-2022 or 2020")
    p.add_argument("--cats", default="cs.LG,cs.AI,stat.ML", help="Comma-separated arXiv categories")
    p.add_argument("--max-per-shard", type=int, default=1000, help="Max results per (year,category) shard")
    p.add_argument("--include-surveys", action="store_true", help="Include survey/review papers")
    p.add_argument("--probe", action="store_true", help="Probe only (count results; do not download)")
    p.add_argument("--delay", type=int, default=3, help="Client delay_seconds between requests")
    p.add_argument("--retries", type=int, default=5, help="Client num_retries")
    p.add_argument("--page-size", type=int, default=50, help="Client page_size")
    return p.parse_args()

def parse_years(s: str) -> List[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s)]

def main():
    args = parse_args()
    years = parse_years(args.years)
    cats = [c.strip() for c in args.cats.split(",") if c.strip()]
    run(
        out_dir=args.out,
        years=years,
        categories=cats,
        max_results_per_shard=args.max_per_shard,
        include_surveys=args.include_surveys,
        probe_only=args.probe,
        delay=args.delay,
        retries=args.retries,
        page_size=args.page_size,
    )

if __name__ == "__main__":
    main()

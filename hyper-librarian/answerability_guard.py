# answerability_guard.py
from __future__ import annotations
import re
from typing import List, Dict, Any
import sqlite3

FINANCE_HINTS = {
    "revenue","expense","profit","loss","income","cash","liquidity","debt","capital",
    "guidance","md&a","mdna","risk","item","10-k","10k","gaap","non-gaap","segment",
    "margin","gross","operating","opex","capex","impairment","currency","fx","hedge",
    "tax","share","dividend","buyback","interest","loan","covenant","goodwill"
}

STOPLIKE = re.compile(r"[^a-z0-9 ]+", re.I)

def _tokens(s: str) -> list[str]:
    s = s.lower()
    s = STOPLIKE.sub(" ", s)
    return [t for t in s.split() if len(t) >= 3]

def _fts_has_token(sq: sqlite3.Connection, tok: str) -> bool:
    # quick existence probe (LIMIT 1) — cheap
    tok = tok.replace("'", "''")
    sql = f"SELECT 1 FROM fts_chunks WHERE fts_chunks MATCH '{tok}' LIMIT 1"
    try:
        row = sq.execute(sql).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False

def assess_answerability(
    q: str,
    picks: List[Dict[str, Any]],
    timings: Dict[str, Any],
    sq: sqlite3.Connection,
    min_overlap_sentences: int = 1,
    min_lex_hits: int = 5,
    min_ann_sim: float = 0.28
) -> (bool, List[str]):
    reasons: List[str] = []
    toks = _tokens(q)

    # R1: ANN similarity too low AND few lexical hits
    ann_sim = float(timings.get("ann_top_sim", 0.0))
    lex_count = int(timings.get("lex_count", 0))
    if ann_sim < min_ann_sim and lex_count < min_lex_hits:
        reasons.append(f"low ANN similarity ({ann_sim:.2f}) and few FTS hits ({lex_count})")

    # R2: No query tokens exist in corpus
    hits = 0
    for t in toks[:8]:  # cap probes
        if _fts_has_token(sq, t):
            hits += 1
    if hits < 2:
        reasons.append(f"only {hits} query tokens found in corpus")

    # R3: No finance/10-K hints in query (unless ANN is strong)
    if ann_sim < (min_ann_sim + 0.05) and not (set(toks) & FINANCE_HINTS):
        reasons.append("query lacks 10-K/finance terms")

    # R4: Evidence overlap: require at least one sentence to share ≥2 tokens with query
    overlap_ok = 0
    qset = set(toks)
    for p in picks:
        ps = set(_tokens(p["text"]))
        if len(qset & ps) >= 2:
            overlap_ok += 1
    if overlap_ok < min_overlap_sentences:
        reasons.append("weak lexical overlap with evidence")

    ok = len(reasons) == 0
    return ok, reasons

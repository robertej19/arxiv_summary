# 41_answer_compose.py
from __future__ import annotations
import re, duckdb
from typing import List, Tuple, Dict, Any

_NUM = re.compile(r"\d")
_BAD = [r"forward-looking statements", r"undue reliance", r"safe harbor",
        r"not constitute", r"website.*incorporated by reference"]
_DRIVERS = {"increase","decrease","grew","declined","rose","fell","driven","due","attributed",
            "impact","headwinds","tailwinds","currency","fx","supply","demand","pricing","margin",
            "cost","volume","mix","risk","inflation","strategy","segments","segment","guidance"}

def _tok(s:str)->list[str]: return re.findall(r"[a-zA-Z0-9]+", s.lower())
def _jac(a:set,b:set)->float:
    if not a or not b: return 0.0
    u=len(a|b);  return len(a&b)/u if u else 0.0

def _score_sentence(sent:str, q_terms:set[str])->float:
    toks=_tok(sent);  n=len(toks)
    if not n: return 0.0
    if any(re.search(p, sent, flags=re.I) for p in _BAD): return -1.0
    length = 1.0 if 8<=n<=35 else (0.3 if 6<=n<=45 else -0.5)
    num    = 1.0 if _NUM.search(sent) else 0.0
    drv    = 0.6 if any(w in _DRIVERS for w in toks) else 0.0
    overlap= len(set(toks)&q_terms)/max(1,len(q_terms))
    return 0.7*overlap + length + num + drv

def select_sentences(
    duck: duckdb.DuckDBPyConnection,
    chunk_rows: List[Tuple],  # (chunk_id,ticker,filing_date,item,text,s0,s1,section_id,doc_id)
    query: str,
    target: int = 4
) -> List[Dict[str,Any]]:
    q_terms=set(_tok(query)); picks:List[Dict[str,Any]]=[]; seen:list[set]=[]
    for (chunk_id,ticker,filing_date,item,_txt,s0,s1,section_id,doc_id) in chunk_rows:
        rows = duck.execute(f"""
            SELECT sent_id, sent_idx, text, sent_hash
            FROM sentences
            WHERE section_id = {int(section_id)} AND sent_idx BETWEEN {int(s0)} AND {int(s1)}
            ORDER BY sent_idx
        """).fetchall()
        for (sent_id, sent_idx, text, sent_hash) in rows:
            text = text.strip()
            sc = _score_sentence(text, q_terms)
            if sc <= 0: continue
            toks=set(_tok(text))
            if any(_jac(toks, prev)>0.8 for prev in seen): continue
            seen.append(toks)
            picks.append({
                "score": sc,
                "text": text,
                "ticker": ticker,
                "year": str(filing_date)[:4] if filing_date else "",
                "item": item,
                "doc_id": int(doc_id),
                "section_id": int(section_id),
                "sent_id": int(sent_id),
                "sent_idx": int(sent_idx),
                "hash": str(sent_hash)[:10],
            })
    picks.sort(key=lambda x:x["score"], reverse=True)
    return picks[:target]

def answerability(picks: List[Dict[str,Any]], min_sources:int=2) -> bool:
    if len(picks)<2: return False
    src={(p["ticker"],p["year"],p["item"]) for p in picks}
    return len(src)>=min_sources

def compose_answer(picks: List[Dict[str,Any]]) -> str:
    if not picks:
        return "I couldn’t find enough grounded statements in the filings to answer this precisely."
    parts=[]
    for p in picks:
        cite=f"[{p['ticker']} {p['year']} • {p['item']} • sent {p['sent_idx']} • {p['hash']}]"
        parts.append(f"{p['text']} {cite}")
    return " ".join(parts)

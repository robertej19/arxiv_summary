# local_corpus/agent_demo.py
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp_model import LlamaCppModel
from tools_local import LocalSearchTool, ReadChunkTool

# =============================================================================
# Environment defaults (from your setup)
# =============================================================================
MODEL_PATH = os.getenv("LLAMA_GGUF", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.getenv("LLAMA_N_CTX", "32768"))
MAX_GEN_TOK = int(os.getenv("LLAMA_MAX_TOK", "800"))

# =============================================================================
# Budgets / Tunables
# =============================================================================
MAX_QUERIES = 2                 # planner: max search queries
K_PER_QUERY = 4                 # search: top-k per query (pre-LLM-rerank)
TOP_AFTER_RERANK = 4            # keep this many hits after LLM rerank
MAX_READ_CHUNKS = 3             # reader: read this many chunks total
NEIGHBOR_WINDOW = 1             # read ±N neighbor chunks
MAX_QUOTES_PER_READ = 2
MAX_CHARS_FROM_READER = 4000
MAX_EVIDENCE_FOR_WRITE = 8
MAX_SENTENCES_OUT = 12          # writer hint

SATURATION_ROUNDS = 1           # extra search->read loops if gaps/contradictions
MIN_UNIQUE_SOURCES = 2

# =============================================================================
# CLI Progress Logging / Spinner
# =============================================================================
VERBOSE = os.getenv("DR_VERBOSE", "1") != "0"   # override with -q/--quiet
SAFE_LOG_MAX = 500

def _s(text: str) -> str:
    text = (text or "").replace("\n", " ")
    return (text[:SAFE_LOG_MAX] + " …") if len(text) > SAFE_LOG_MAX else text

class Log:
    @staticmethod
    def info(msg, *args):
        if VERBOSE:
            print("•", msg % args if args else msg)
            sys.stdout.flush()

class Spinner:
    def __init__(self, label="working"):
        self.label = label
        self._stop = Event()
        self._t: Optional[Thread] = None

    def start(self):
        if not VERBOSE: return
        def run():
            chars = "|/-\\"
            i = 0
            while not self._stop.is_set():
                sys.stdout.write(f"\r{self.label} {chars[i % 4]}")
                sys.stdout.flush()
                i += 1
                time.sleep(0.08)
            sys.stdout.write("\r" + " " * (len(self.label) + 2) + "\r")
            sys.stdout.flush()
        self._t = Thread(target=run, daemon=True)
        self._t.start()

    def stop(self):
        if not VERBOSE: return
        self._stop.set()
        if self._t:
            self._t.join()

# =============================================================================
# Data structures
# =============================================================================
@dataclass
class Evidence:
    doc_id: str
    chunk_id: str
    title: str
    quote: str
    year: Optional[str] = None
    section: Optional[str] = None
    score: float = 0.0
    span: Optional[Tuple[int, int]] = None  # approx char offsets within read text

@dataclass
class Notepad:
    notes: List[Evidence] = field(default_factory=list)

    def add(self, ev: Evidence):
        key = (ev.doc_id, ev.chunk_id, ev.quote[:160])
        if key not in {(n.doc_id, n.chunk_id, n.quote[:160]) for n in self.notes}:
            self.notes.append(ev)

    def unique_sources(self) -> List[Tuple[str, str]]:
        seen = set()
        out: List[Tuple[str, str]] = []
        for n in self.notes:
            if n.doc_id not in seen:
                seen.add(n.doc_id)
                out.append((n.doc_id, n.title or n.doc_id))
        return out

    def unique_source_count(self) -> int:
        return len({n.doc_id for n in self.notes})

    def select_for_keywords(self, keywords: List[str], limit: int) -> List[Evidence]:
        ranked = []
        for n in self.notes:
            hit = sum(kw.lower() in n.quote.lower() for kw in keywords)
            ranked.append((hit + n.score / 10.0, n))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in ranked[:limit]]

# =============================================================================
# Light helpers
# =============================================================================
STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its of on or that the their there these this to was were what when where which who why will with
""".split())

def _keywords(text: str, max_k: int = 10) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    seen = set(); out = []
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= max_k:
            break
    return out or toks[:max_k]

def _extract_json_array(s: str) -> List[str]:
    m = re.search(r"\[[\s\S]*?\]", s or "")
    if not m: return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr if isinstance(x, str)]
    except Exception:
        return []

def _safe_head(s: str, n: int) -> str:
    s = (s or "").replace("\u0000", "")
    return s if len(s) <= n else s[:n] + " …[truncated]"

def _find_span(hay: str, needle: str) -> Optional[Tuple[int, int]]:
    i = (hay or "").find(needle or "")
    if i < 0:
        # Try normalized search; if found, skip offsets (avoid false precision)
        def norm(t: str) -> str:
            return re.sub(r"\s+", " ", t).strip()
        H, N = norm(hay or ""), norm(needle or "")
        return None if H.find(N) < 0 else None
    return (i, i + len(needle))

# =============================================================================
# Model & tools
# =============================================================================
def build_model() -> LlamaCppModel:
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    return LlamaCppModel(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        max_tokens=MAX_GEN_TOK,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        verbose=False,
    )

def build_tools() -> Tuple[LocalSearchTool, ReadChunkTool]:
    return LocalSearchTool(), ReadChunkTool()

# =============================================================================
# Prompts
# =============================================================================
PLANNER_PROMPT = """You are a research planner.
Generate AT MOST 2 very specific search queries for the user's question.
Return ONLY a JSON array of strings. No notes.

Question:
{question}
"""

CLAIMS_PROMPT = """You are an analyst.
Break the question into 4–8 short, atomic claims that, if supported by sources, would fully answer it.
Each claim must be a single sentence, specific, and testable (not vague).
Return ONLY a JSON array of strings. No commentary.

Question:
{question}
"""

RERANK_PROMPT = """Score the DIRECT relevance of each snippet to the query on a 0–3 scale:
0=no, 1=weak, 2=good, 3=excellent.
Return ONLY a JSON array of integers (one per item, same order). No extra text.

Query: "{query}"

Snippets:
{snippets}
"""

WRITER_PROMPT = """You are a careful technical writer.
You must answer using ONLY the numbered evidence quotes below. Do not invent facts.

Question:
{question}

Claims to cover (each should be supported with at least one citation):
{claims_bullets}

Evidence (numbered):
{evidence_blobs}

Citations:
{citation_list}

Instructions:
- Write {max_sentences} concise sentences or bullets total.
- Cover all listed claims; if a claim is not supported by the evidence, clearly flag it as "unsupported".
- Attach citation markers like [1], [2] immediately after the sentences they support.
- Prefer precise language, avoid fluff, and keep it self-contained.
"""

CRITIC_PROMPT = """You are an auditor.
Given the draft answer, the list of claims, and the evidence mapping, check:
1) Does EACH claim have at least one supporting citation [n] in the draft?
2) Are there any obvious contradictions between cited quotes within the same claim?
Return a JSON object with:
{
  "supported_claims": [indexes],
  "unsupported_claims": [indexes],
  "possible_contradictions": [{"claim_index": i, "notes": "..."}]
}
Only return JSON, no commentary.

Draft:
{draft}

Claims:
{claims_bullets}

Evidence map (doc numbers correspond to [n] in draft):
{evidence_map}
"""

CONTRADICT_PROMPT = """Do these two quotes contradict each other about the same fact? Answer with a single word: Yes or No.

Quote A: "{qa}"
Quote B: "{qb}"
"""

# =============================================================================
# Pipeline stages
# =============================================================================
def plan_queries(model: LlamaCppModel, question: str) -> List[str]:
    Log.info("Planning queries…")
    sp = Spinner("planning"); sp.start()
    out = model.generate([{"role": "user", "content": PLANNER_PROMPT.format(question=question)}]).content
    sp.stop()
    arr = _extract_json_array(out)[:MAX_QUERIES]
    Log.info("Queries: %s", arr or [question])
    return arr or [question]

def extract_claims(model: LlamaCppModel, question: str) -> List[str]:
    Log.info("Extracting claims…")
    sp = Spinner("claims"); sp.start()
    out = model.generate([{"role": "user", "content": CLAIMS_PROMPT.format(question=question)}]).content
    sp.stop()
    claims = [c.strip() for c in _extract_json_array(out)]
    if len(claims) < 4:
        Log.info("Few claims from model; using keyword fallback.")
        kws = _keywords(question, max_k=6)
        claims = [f"The answer should address: {kw}" for kw in kws][:4]
    claims = claims[:8]
    Log.info("Claims (%d): %s", len(claims), claims)
    return claims

def rerank_hits_with_llm(model: LlamaCppModel, query: str, hits: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    Log.info("Reranking %d hits…", len(hits))
    lines = []
    for i, h in enumerate(hits[:10], 1):
        snip = _safe_head(h.get("snippet", "") or "", 280)
        lines.append(f"{i}. ({h.get('doc_id')}::{h.get('chunk_id')}) {snip}")
    prompt = RERANK_PROMPT.format(query=query, snippets="\n".join(lines))
    sp = Spinner("reranking"); sp.start()
    resp = model.generate([{"role": "user", "content": prompt}]).content
    sp.stop()
    try:
        arr = json.loads(re.search(r"\[[\s\S]*\]", resp).group(0))
        scores = [int(x) for x in arr]
    except Exception:
        scores = [1] * min(10, len(hits))
    pairs = list(zip(hits[:len(scores)], scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    kept = [h for h, _ in pairs[:top_n]]
    Log.info("Kept top %d after rerank.", len(kept))
    return kept

def read_and_quote(read_tool: ReadChunkTool, hit: Dict[str, Any], keywords: List[str]) -> List[Evidence]:
    Log.info("Reading %s::%s (±%d)…", hit.get("doc_id"), hit.get("chunk_id"), NEIGHBOR_WINDOW)
    evidences: List[Evidence] = []
    # Quotes
    qres = read_tool.forward(
        chunk_id=hit["chunk_id"],
        include_neighbors=NEIGHBOR_WINDOW,
        mode="quotes",
        keywords=keywords,
        max_quotes=MAX_QUOTES_PER_READ,
        max_chars=MAX_CHARS_FROM_READER,
    )
    quotes = qres.get("quotes") or []
    citation = qres.get("citation") or {}
    # Full text for span estimation (not sent to model)
    tres = read_tool.forward(
        chunk_id=hit["chunk_id"],
        include_neighbors=NEIGHBOR_WINDOW,
        mode="text",
        max_chars=MAX_CHARS_FROM_READER,
    )
    full_text = tres.get("text") or ""
    for qt in quotes:
        span = _find_span(full_text, qt)
        evidences.append(Evidence(
            doc_id=citation.get("doc_id") or hit.get("doc_id"),
            chunk_id=citation.get("chunk_id") or hit.get("chunk_id"),
            title=citation.get("title") or hit.get("title") or "",
            year=citation.get("year") or hit.get("year"),
            quote=qt,
            score=float(hit.get("score", 0.0)),
            section=hit.get("section"),
            span=span
        ))
    Log.info("  → extracted %d quote(s).", len(evidences))
    return evidences

def detect_numeric_conflict(q1: str, q2: str) -> bool:
    nums1 = re.findall(r"-?\d+(?:\.\d+)?", q1 or "")
    nums2 = re.findall(r"-?\d+(?:\.\d+)?", q2 or "")
    return bool(nums1 and nums2 and set(nums1) != set(nums2))

def llm_contradiction_check(model: LlamaCppModel, qa: str, qb: str) -> bool:
    resp = model.generate([{"role": "user", "content": CONTRADICT_PROMPT.format(qa=_safe_head(qa, 400), qb=_safe_head(qb, 400))}]).content.strip()
    return resp.lower().startswith("y")

def map_claims_to_evidence(claims: List[str], pad: Notepad) -> Dict[int, List[Evidence]]:
    mapping: Dict[int, List[Evidence]] = {}
    for i, claim in enumerate(claims):
        kws = _keywords(claim, max_k=8)
        chosen = pad.select_for_keywords(kws, limit=3)
        mapping[i] = chosen
    return mapping

def contradictions_for_claim(model: LlamaCppModel, evs: List[Evidence]) -> List[str]:
    issues = []
    for i in range(len(evs)):
        for j in range(i + 1, len(evs)):
            a, b = evs[i], evs[j]
            if detect_numeric_conflict(a.quote, b.quote):
                if llm_contradiction_check(model, a.quote, b.quote):
                    issues.append(f"{a.doc_id}::{a.chunk_id} vs {b.doc_id}::{b.chunk_id}")
    return issues

# =============================================================================
# Writer & Critic
# =============================================================================
def build_evidence_pack(pad: Notepad, claims: List[str], keywords: List[str]) -> Tuple[str, str, Dict[str, int], List[Evidence]]:
    focus = pad.select_for_keywords(keywords, limit=MAX_EVIDENCE_FOR_WRITE)
    numbered_sources: List[Tuple[str, str]] = []
    seen = set()
    for ev in focus:
        if ev.doc_id not in seen:
            seen.add(ev.doc_id)
            numbered_sources.append((ev.doc_id, ev.title or ev.doc_id))
    for doc_id, title in pad.unique_sources():
        if doc_id not in seen:
            seen.add(doc_id)
            numbered_sources.append((doc_id, title))
        if len(numbered_sources) >= 12:
            break
    idx_of_doc = {doc: i + 1 for i, (doc, _title) in enumerate(numbered_sources)}
    blobs = []
    for ev in focus:
        marker = idx_of_doc.get(ev.doc_id, 1)
        loc = f"{ev.doc_id}::{ev.chunk_id}"
        if ev.span:
            loc += f" [chars {ev.span[0]}–{ev.span[1]}]"
        sec = f" — {ev.section}" if ev.section else ""
        blobs.append(f"[{marker}] {ev.quote}\n(source: {loc}{sec})")
    evidence_blobs = "\n\n".join(blobs)
    citation_list = "\n".join([f"- [{i+1}] {title} (doc_id={doc})" for i, (doc, title) in enumerate(numbered_sources)])
    return evidence_blobs, citation_list, idx_of_doc, focus

def write_answer(model: LlamaCppModel, question: str, claims: List[str],
                 evidence_blobs: str, citation_list: str) -> str:
    claims_bullets = "\n".join([f"- ({i+1}) {c}" for i, c in enumerate(claims)])
    prompt = WRITER_PROMPT.format(
        question=question,
        claims_bullets=claims_bullets,
        evidence_blobs=evidence_blobs,
        citation_list=citation_list,
        max_sentences=MAX_SENTENCES_OUT
    )
    return model.generate([{"role": "user", "content": prompt}]).content

def critic_pass(model: LlamaCppModel, draft: str, claims: List[str],
                claim_to_evidence: Dict[int, List[Evidence]], idx_of_doc: Dict[str, int]) -> Dict[str, Any]:
    claims_bullets = "\n".join([f"- ({i+1}) {c}" for i, c in enumerate(claims)])
    evidence_map_lines = [f"[{idx}] {doc}" for doc, idx in idx_of_doc.items()]
    prompt = CRITIC_PROMPT.format(
        draft=_safe_head(draft, 6500),
        claims_bullets=claims_bullets,
        evidence_map="\n".join(evidence_map_lines),
    )
    sp = Spinner("critic"); sp.start()
    resp = model.generate([{"role": "user", "content": prompt}]).content
    sp.stop()
    try:
        obj = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        obj = {"supported_claims": [], "unsupported_claims": list(range(1, len(claims) + 1)), "possible_contradictions": []}
    return obj

# =============================================================================
# Main pipeline
# =============================================================================
def deep_research(question: str) -> str:
    Log.info("Question: %s", _s(question))
    model = build_model()
    search_tool, read_tool = build_tools()

    # PLAN & CLAIMS
    queries = plan_queries(model, question)
    claims = extract_claims(model, question)
    q_keywords = _keywords(question, max_k=10)

    # SEARCH
    Log.info("Searching (k=%d per query)…", K_PER_QUERY)
    seen = set(); raw_hits: List[Dict[str, Any]] = []
    for q in queries:
        res = search_tool.forward(query=q, k=K_PER_QUERY)
        items = res.get("results", []) if isinstance(res, dict) else []
        for it in items:
            key = (it.get("doc_id"), it.get("chunk_id"))
            if key not in seen:
                raw_hits.append(it); seen.add(key)
    Log.info("Raw hits (unique): %d", len(raw_hits))

    # RERANK
    reranked = rerank_hits_with_llm(model, query=question, hits=raw_hits, top_n=TOP_AFTER_RERANK)
    Log.info("Reranked top: %d", len(reranked))

    # READ
    pad = Notepad()
    for hit in reranked[:MAX_READ_CHUNKS]:
        for ev in read_and_quote(read_tool, hit, q_keywords):
            pad.add(ev)
    Log.info("Evidence collected: %d item(s) from %d source(s)", len(pad.notes), pad.unique_source_count())

    # SATURATION / GUARDRAIL
    rounds_left = SATURATION_ROUNDS
    if pad.unique_source_count() < MIN_UNIQUE_SOURCES and rounds_left > 0:
        Log.info("Insufficient sources, attempting saturation round…")
        rounds_left -= 1
        expanded = " ".join(_keywords(" ".join([e.quote for e in pad.notes]), max_k=6)) or question
        res = search_tool.forward(query=expanded, k=K_PER_QUERY)
        items = res.get("results", []) if isinstance(res, dict) else []
        for it in items:
            if pad.unique_source_count() >= MIN_UNIQUE_SOURCES:
                break
            if any((n.doc_id == it["doc_id"] and n.chunk_id == it["chunk_id"]) for n in pad.notes):
                continue
            for ev in read_and_quote(read_tool, it, q_keywords):
                pad.add(ev)
        Log.info("After saturation: %d source(s), %d evidence", pad.unique_source_count(), len(pad.notes))

    if pad.unique_source_count() < MIN_UNIQUE_SOURCES:
        Log.info("Still insufficient sources. Aborting with message.")
        return ("Insufficient evidence from local corpus (need ≥2 distinct sources). "
                "Consider expanding the corpus or relaxing constraints.")

    # CLAIM MAPPING & CONTRADICTIONS
    Log.info("Mapping evidence to claims…")
    claim_map = map_claims_to_evidence(claims, pad)
    contradictions: List[Tuple[int, List[str]]] = []
    for i, evs in claim_map.items():
        if len(evs) >= 2:
            issues = contradictions_for_claim(model, evs[:3])
            if issues:
                Log.info("  Possible contradiction in claim (%d): %s", i+1, issues)
                contradictions.append((i, issues))
    if not contradictions:
        Log.info("No contradictions detected in mapped evidence.")

    # WRITE
    Log.info("Composing draft answer…")
    evidence_blobs, citation_list, idx_of_doc, focus = build_evidence_pack(pad, claims, q_keywords)
    sp = Spinner("writing"); sp.start()
    draft = write_answer(model, question, claims, evidence_blobs, citation_list)
    sp.stop()
    Log.info("Draft length: %d chars", len(draft))

    # CRITIC
    Log.info("Running critic pass…")
    critic = critic_pass(model, draft, claims, claim_map, idx_of_doc)
    supported = set(int(i) for i in critic.get("supported_claims", []))
    unsupported = set(int(i) for i in critic.get("unsupported_claims", []))
    contradictions_report = critic.get("possible_contradictions", []) or []
    Log.info("Coverage: %d/%d supported. Unsupported: %s", len(supported), len(claims), sorted(unsupported))

    # QUALITY FOOTER
    total_claims = len(claims)
    covered = len(supported)
    coverage_pct = int(round(100.0 * covered / max(1, total_claims)))
    contradict_count = len(contradictions) + len(contradictions_report)
    conf = "high" if coverage_pct >= 90 and contradict_count == 0 and pad.unique_source_count() >= 3 else \
           "medium" if coverage_pct >= 70 and contradict_count <= 1 else "low"

    lines = []
    lines.append("╭──────────────────────────────── Answer ────────────────────────────────╮")
    lines.append(draft.strip())
    lines.append("╰────────────────────────────────────────────────────────────────────────╯\n")

    lines.append("— Research summary —")
    lines.append(f"Sources used: {pad.unique_source_count()}  |  Evidence items: {len(pad.notes)}")
    lines.append(f"Claims covered: {covered}/{total_claims} ({coverage_pct}%)")
    if unsupported:
        miss = ", ".join(str(i) for i in sorted(unsupported))
        lines.append(f"Unsupported claims: {miss}")
    if contradict_count:
        lines.append(f"Possible contradictions: {contradict_count}")
        for i, issues in contradictions:
            lines.append(f"  • Claim ({i+1}) conflicts: {', '.join(issues)}")
        for item in contradictions_report:
            lines.append(f"  • Critic flagged claim ({item.get('claim_index')}) — {item.get('notes')}")

    lines.append("Sources:")
    for idx, (doc, title) in enumerate(pad.unique_sources(), 1):
        lines.append(f"  [{idx}] {title} (doc_id={doc})")

    lines.append(f"Confidence: {conf}")
    return "\n".join(lines)

# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Research question")
    parser.add_argument("--quiet", "-q", action="store_true", help="Silence progress logs")
    args = parser.parse_args()

    global VERBOSE
    if args.quiet:
        VERBOSE = False

    question = " ".join(args.question) if args.question else \
        "Write a short report on the different policy optimization algorithms, for example PPO, GRPO, etc."

    print("╭──────────────────────────────── Deep Research ────────────────────────────────╮")
    print("│ Question:", question)
    print("╰──────────────────────────────────────────────────────────────────────────────╯")

    answer = deep_research(question)
    print(answer)

if __name__ == "__main__":
    if "Qwen2.5-7B-Instruct" not in MODEL_PATH and not os.path.exists(MODEL_PATH):
        print(f"WARNING: MODEL_PATH may be incorrect. Current: {MODEL_PATH}", file=sys.stderr)
    main()

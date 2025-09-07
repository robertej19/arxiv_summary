# local_corpus/agent_demo.py
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Local model + tools ---
from llama_cpp_model import LlamaCppModel
from tools_local import LocalSearchTool, ReadChunkTool

# --------------------------
# Config (tune as you like)
# --------------------------
MODEL_PATH = os.getenv("LLAMA_GGUF", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")  # set env var or edit here
N_CTX = int(os.getenv("LLAMA_N_CTX", "32768"))                     # raise to 32768 if your GGUF/VRAM allow
MAX_GEN_TOK = int(os.getenv("LLAMA_MAX_TOK", "800"))

MAX_QUERIES = 2           # planner: at most 2 focused queries
K_PER_QUERY = 3           # search: top k per query
MAX_READ_CHUNKS = 3       # reader: read at most 3 chunks total
FALLBACK_EXTRA_QUERIES = 1
MAX_QUOTES_PER_READ = 2   # reader: quotes to extract per chunk
NEIGHBOR_WINDOW = 1       # read ±1 neighbor chunk for context
MAX_EVIDENCE_FOR_WRITE = 6

# --------------------------
# Evidence Scratchpad
# --------------------------
@dataclass
class Evidence:
    doc_id: str
    chunk_id: str
    title: str
    quote: str
    year: Optional[str] = None
    section: Optional[str] = None
    score: float = 0.0

@dataclass
class Notepad:
    notes: List[Evidence] = field(default_factory=list)

    def add(self, ev: Evidence):
        key = (ev.doc_id, ev.chunk_id, ev.quote[:120])
        if key not in {(n.doc_id, n.chunk_id, n.quote[:120]) for n in self.notes}:
            self.notes.append(ev)

    def unique_sources(self) -> List[Tuple[str, str]]:
        """Return [(doc_id, title)] in order of first appearance."""
        seen = set()
        out: List[Tuple[str, str]] = []
        for n in self.notes:
            if n.doc_id not in seen:
                seen.add(n.doc_id)
                out.append((n.doc_id, n.title or n.doc_id))
        return out

    def unique_source_count(self) -> int:
        return len({n.doc_id for n in self.notes})

    def select_for_claims(self, keywords: List[str], limit: int = MAX_EVIDENCE_FOR_WRITE) -> List[Evidence]:
        # rank notes by keyword overlap + score
        ranked = []
        for n in self.notes:
            hit = sum(kw.lower() in n.quote.lower() for kw in keywords)
            ranked.append((hit + n.score / 10.0, n))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in ranked[:limit]]

# --------------------------
# Tiny helpers
# --------------------------
STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its of on or that the their there these this to was were what when where which who why will with
""".split())

def _keywords_from_question(q: str, max_k: int = 8) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", q.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    # keep order, dedupe
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_k:
            break
    return out or toks[:max_k]

def _extract_json_strings(s: str) -> List[str]:
    """
    Find the first JSON array of strings in s, return it.
    Accepts bare JSON or code fences. Fallback: split by newline.
    """
    # try to find a JSON array
    m = re.search(r"\[\s*(\".*?\")\s*(,.*)?\]", s, flags=re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x) for x in arr if isinstance(x, str)]
        except Exception:
            pass
    # fallback: take lines that look like queries
    lines = [ln.strip().strip("-• ").strip() for ln in s.splitlines() if ln.strip()]
    # remove code fences if any
    lines = [ln for ln in lines if not ln.startswith("```") and not ln.endswith("```")]
    # heuristics: keep 1–2 reasonable lines
    return lines[:2] if lines else []

def _parse_results(agent_text: str) -> List[Dict[str, Any]]:
    """
    Try to parse tool output that might have been echoed by an agent.
    We accept either a dict-like string or inline JSON.
    """
    # extract a {"results": [...]} JSON object if present
    m = re.search(r"\{[\s\S]*?\"results\"\s*:\s*\[[\s\S]*?\][\s\S]*?\}", agent_text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
                return obj["results"]
        except Exception:
            pass
    # else, nothing parsable
    return []

def _extract_key_quotes(text: str, keywords: List[str], max_quotes: int) -> List[str]:
    # light sentence split + keyword scoring (fallback if the tool returns full text)
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) >= 30]
    scored = []
    for s in sents:
        score = sum(kw.lower() in s.lower() for kw in keywords)
        length_bonus = 1.0 - abs(len(s) - 180) / 300.0
        scored.append((score + max(0.0, length_bonus), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored[:max_quotes]]

def _expand_query_from_notes(pad: Notepad, fallback: str) -> str:
    # Build a simple expansion using high-signal terms in notes; else fallback.
    txt = " ".join(ev.quote for ev in pad.notes[:4])
    kws = _keywords_from_question(txt, max_k=6)
    return " ".join(kws) if kws else fallback

# --------------------------
# Model + Tools
# --------------------------
def build_model() -> LlamaCppModel:
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MODEL_PATH not found: {MODEL_PATH}", file=sys.stderr)
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

# --------------------------
# Planner / Writer prompts
# --------------------------
PLANNER_PROMPT = """You are a focused research planner.
Task: Generate AT MOST 2 highly targeted search queries to answer the user question.
Return ONLY a JSON array of strings. No commentary.

Question:
{question}
"""

WRITER_PROMPT = """You are a careful technical writer. Use ONLY the evidence snippets below to answer the question.
Attach citation markers like [1], [2] right after the sentences they support. Do not invent information.

Question:
{question}

Evidence (numbered):
{evidence_blobs}

Citations:
{citation_list}

Write a concise answer (5–10 sentences or bullet points). Avoid fluff. Use simple language and be precise.
"""

# --------------------------
# Deep research pipeline
# --------------------------
def deep_research(question: str) -> str:
    model = build_model()
    search_tool, read_tool = build_tools()
    pad = Notepad()

    # 1) PLAN → up to 2 queries
    planner_out = model.generate([{"role": "user", "content": PLANNER_PROMPT.format(question=question)}]).content
    queries = _extract_json_strings(planner_out)[:MAX_QUERIES]
    if not queries:
        queries = [question]

    # 2) SEARCH (union of hits; no giant logs)
    seen_keys = set()
    hits: List[Dict[str, Any]] = []
    for q in queries:
        res = search_tool.forward(query=q, k=K_PER_QUERY)
        items = res.get("results", []) if isinstance(res, dict) else []
        for it in items:
            key = (it.get("doc_id"), it.get("chunk_id"))
            if key not in seen_keys:
                seen_keys.add(key)
                hits.append(it)
        if len(hits) >= MAX_READ_CHUNKS * 2:  # small buffer
            break

    # 3) READ top chunks → quotes into Notepad
    keywords = _keywords_from_question(question)
    for it in hits[:MAX_READ_CHUNKS]:
        cid = it["chunk_id"]
        read = read_tool.forward(
            chunk_id=cid,
            include_neighbors=NEIGHBOR_WINDOW,
            mode="quotes",
            keywords=keywords,
            max_quotes=MAX_QUOTES_PER_READ,
        )
        quotes = read.get("quotes") or []
        citation = read.get("citation") or {}
        for qt in quotes:
            pad.add(Evidence(
                doc_id=citation.get("doc_id") or it.get("doc_id"),
                chunk_id=citation.get("chunk_id") or it.get("chunk_id"),
                title=citation.get("title") or it.get("title") or "",
                year=citation.get("year") or it.get("year"),
                quote=qt,
                score=float(it.get("score", 0.0)),
                section=it.get("section"),
            ))

    # 3b) Guardrail: ensure ≥2 unique sources, else 1 fallback query & read
    if pad.unique_source_count() < 2:
        extra_q = _expand_query_from_notes(pad, fallback=question)
        res2 = search_tool.forward(query=extra_q, k=K_PER_QUERY)
        items2 = res2.get("results", []) if isinstance(res2, dict) else []
        for it in items2:
            if pad.unique_source_count() >= 2:
                break
            cid = it["chunk_id"]
            read = read_tool.forward(
                chunk_id=cid,
                include_neighbors=NEIGHBOR_WINDOW,
                mode="quotes",
                keywords=keywords,
                max_quotes=1,
            )
            quotes = read.get("quotes") or []
            citation = read.get("citation") or {}
            for qt in quotes:
                pad.add(Evidence(
                    doc_id=citation.get("doc_id") or it.get("doc_id"),
                    chunk_id=citation.get("chunk_id") or it.get("chunk_id"),
                    title=citation.get("title") or it.get("title") or "",
                    year=citation.get("year") or it.get("year"),
                    quote=qt,
                    score=float(it.get("score", 0.0)),
                    section=it.get("section"),
                ))

    if pad.unique_source_count() < 2:
        return ("Insufficient evidence from local corpus (need ≥2 distinct sources). "
                "Consider expanding the corpus or relaxing constraints.")

    # 4) PREPARE EVIDENCE for writer
    focus = pad.select_for_claims(keywords, limit=MAX_EVIDENCE_FOR_WRITE)
    # map first appearance of each (doc_id) within focus to [number]
    numbered_sources: List[Tuple[str, str]] = []
    seen_docs = set()
    for ev in focus:
        if ev.doc_id not in seen_docs:
            seen_docs.add(ev.doc_id)
            numbered_sources.append((ev.doc_id, ev.title or ev.doc_id))
    # if still missing some sources, append from overall pad
    for doc_id, title in pad.unique_sources():
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            numbered_sources.append((doc_id, title))
        if len(numbered_sources) >= 10:
            break

    idx_of_doc = {doc: i + 1 for i, (doc, _title) in enumerate(numbered_sources)}
    evidence_blobs = []
    for ev in focus:
        idx = idx_of_doc.get(ev.doc_id, 1)
        loc = f"{ev.doc_id}::{ev.chunk_id}"
        evidence_blobs.append(f"[{idx}] {ev.quote}\n(source: {loc}{' — '+ev.section if ev.section else ''})")
    citation_list = "\n".join([f"- [{i+1}] {title} (doc_id={doc})" for i, (doc, title) in enumerate(numbered_sources)])

    # 5) WRITE
    writer_prompt = WRITER_PROMPT.format(
        question=question,
        evidence_blobs="\n\n".join(evidence_blobs),
        citation_list=citation_list
    )
    final = model.generate([{"role": "user", "content": writer_prompt}]).content
    return final

# --------------------------
# CLI
# --------------------------
def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Tell me about the different Policy Optimization algorithms, for example PPO, GRPO, etc."

    print("╭───────────────────────────────────────────────────────────────────── Deep Research ──────────────────────────────────────────────────────────────────────╮")
    print("│ Question:", question)
    print("╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯")

    answer = deep_research(question)
    print("\n--- FINAL ANSWER ---\n")
    print(answer)

if __name__ == "__main__":
    # Basic env sanity
    if "path/to/your/model.gguf" in MODEL_PATH:
        print("WARNING: Set LLAMA_GGUF=/abs/path/to/your.gguf (or edit MODEL_PATH in this file).", file=sys.stderr)
    main()

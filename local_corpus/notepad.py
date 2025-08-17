# notepad.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Evidence:
    doc_id: str
    chunk_id: str
    title: str
    quote: str
    span: Optional[str] = None  # "L120-142" or char offsets
    score: float = 0.0

@dataclass
class Notepad:
    notes: List[Evidence] = field(default_factory=list)

    def add(self, ev: Evidence):
        # dedupe by (doc_id, chunk_id, quote[:120])
        key = (ev.doc_id, ev.chunk_id, ev.quote[:120])
        if key not in {(n.doc_id, n.chunk_id, n.quote[:120]) for n in self.notes}:
            self.notes.append(ev)

    def sources(self) -> Dict[str, str]:
        # doc_id -> title
        acc = {}
        for n in self.notes:
            acc[n.doc_id] = getattr(n, "title", n.doc_id)
        return acc

    def unique_source_count(self) -> int:
        return len({n.doc_id for n in self.notes})

    def select_for_claims(self, claim_keywords: List[str], max_evidence=6) -> List[Evidence]:
        # simple keyword re-rank on collected notes
        scored = []
        for n in self.notes:
            hit = sum(kw.lower() in n.quote.lower() for kw in claim_keywords)
            scored.append((hit + n.score/10.0, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:max_evidence]]

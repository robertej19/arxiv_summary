# local_corpus/search_demo.py
import os, json, pickle, re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

STORE_DIR = "./store"
CHUNKS = os.path.join(STORE_DIR, "chunks.jsonl")
BM25_PKL = os.path.join(STORE_DIR, "bm25.pkl")
TOK_PKL = os.path.join(STORE_DIR, "tokens.pkl")
FAISS_INDEX = os.path.join(STORE_DIR, "faiss.index")
IDS_JSON = os.path.join(STORE_DIR, "chunk_ids.json")

EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

def load_all():
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    with open(TOK_PKL, "rb") as f:
        toks = pickle.load(f)
    with open(IDS_JSON, "r", encoding="utf-8") as f:
        ids = json.load(f)
    index = faiss.read_index(FAISS_INDEX)
    # quick random access to chunk text
    id2text = {}
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id2text[rec["chunk_id"]] = rec
    return bm25, toks, index, ids, id2text

def hybrid_search(query, top_k_bm25=200, top_k_vec=200, final_k=20):
    bm25, toks, index, ids, id2text = load_all()
    # BM25
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k_bm25]

    # Vector
    emb_model = SentenceTransformer(EMB_MODEL, device="cpu")
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), top_k_vec)
    vec_top = I[0]

    # union & preliminary score
    cand_idx = list(set(bm25_top.tolist()) | set(vec_top.tolist()))
    cand = [(i, float(bm25_scores[i])) for i in cand_idx]

    # Rerank cross-encoder
    rer = CrossEncoder(RERANK_MODEL, device="cpu")
    pairs = [(query, id2text[ids[i]]["text"][:2000]) for i, _ in cand]
    scores = rer.predict(pairs)

    # sort by reranker
    ranked = sorted(zip(cand_idx, scores), key=lambda x: x[1], reverse=True)[:final_k]
    results = []
    for i, sc in ranked:
        rec = id2text[ids[i]]
        snippet = rec["text"][:400].replace("\n", " ")
        results.append({
            "chunk_id": rec["chunk_id"],
            "doc_id": rec["doc_id"],
            "title": rec["title"],
            "year": rec["year"],
            "score": float(sc),
            "snippet": snippet,
        })
    return results

if __name__ == "__main__":
    q = "How does PPO differ from TRPO and why is it more practical?"
    hits = hybrid_search(q)
    for h in hits[:5]:
        print(f"- {h['title']} ({h['year']}) [{h['doc_id']}]  score={h['score']:.3f}")
        print(f"  {h['snippet']}…")

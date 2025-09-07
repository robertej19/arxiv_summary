# local_corpus/build_index.py
import os, json, pickle, re
from tqdm import tqdm
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

STORE_DIR = "./store"
CHUNKS = os.path.join(STORE_DIR, "chunks.jsonl")
BM25_PKL = os.path.join(STORE_DIR, "bm25.pkl")
TOK_PKL = os.path.join(STORE_DIR, "tokens.pkl")
FAISS_INDEX = os.path.join(STORE_DIR, "faiss.index")
EMB_NPY = os.path.join(STORE_DIR, "embeddings.npy")
IDS_JSON = os.path.join(STORE_DIR, "chunk_ids.json")

EMB_MODEL = "BAAI/bge-small-en-v1.5"   # cached earlier

def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

def load_chunks():
    texts, ids = [], []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            ids.append(rec["chunk_id"])
    return ids, texts

def build_bm25(texts):
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def embed_all(texts, batch=64):
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    # bge uses cosine; normalize for IP in FAISS
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def build_faiss(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
    index.add(embs)
    return index

def run():
    ids, texts = load_chunks()
    print(f"[index] loaded {len(texts)} chunks")

    print("[index] building BM25 …")
    bm25, toks = build_bm25(texts)
    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)
    with open(TOK_PKL, "wb") as f:
        pickle.dump(toks, f)

    print("[index] embedding chunks …")
    embs = embed_all(texts)
    np.save(EMB_NPY, embs)

    print("[index] building FAISS …")
    index = build_faiss(embs)
    faiss.write_index(index, FAISS_INDEX)

    with open(IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print("[index] done.")

if __name__ == "__main__":
    run()

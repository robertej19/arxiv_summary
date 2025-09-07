# streamlit_app.py
from __future__ import annotations
import os
import time
import streamlit as st

# Local modules (make sure files are named exactly like this)
from warm_retriever import WarmRetriever
from answer_compose import select_sentences, answerability, compose_answer

# ----------------------------
# Config panel (left sidebar)
# ----------------------------
st.set_page_config(page_title="Hyper Librarian", page_icon="🔎", layout="wide")

with st.sidebar:
    st.title("🔧 Settings")
    duckdb_path = st.text_input("DuckDB", value=os.environ.get("HL_DUCKDB", "tenk.duckdb"))
    fts_path    = st.text_input("FTS5 SQLite", value=os.environ.get("HL_FTS", "tenk_fts.sqlite"))
    hnsw_path   = st.text_input("HNSW index", value=os.environ.get("HL_HNSW", "tenk_hnsw.bin"))
    ids_npy     = st.text_input("chunk_ids.npy", value=os.environ.get("HL_IDS", "chunk_ids.npy"))
    embs_npy    = st.text_input("chunk_embs.npy", value=os.environ.get("HL_EMBS", "chunk_embs.npy"))
    model_name  = st.text_input("Encoder", value=os.environ.get("HL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    st.markdown("---")
    st.caption("Retrieval knobs")
    k_lex      = st.slider("k_lex (BM25)", 10, 200, 80, 5)
    k_ann      = st.slider("k_ann (ANN)",  10, 200, 80, 5)
    fuse_k     = st.slider("fuse_k (pool)", 16, 200, 48, 4)
    k_final    = st.slider("k_final (MMR picks)", 2, 20, 8, 1)
    mmr_lambda = st.slider("MMR λ (relevance vs diversity)", 0.0, 1.0, 0.5, 0.05)
    target_sents = st.slider("Sentences in answer", 1, 6, 4, 1)
    min_sources  = st.slider("Min distinct sources", 1, 4, 2, 1)

# ----------------------------------------------------
# Warm, cached singletons (kept across Streamlit runs)
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_retriever(
    duckdb_path, fts_path, hnsw_path, ids_npy, embs_npy, model_name,
    k_lex, k_ann, fuse_k, k_final, mmr_lambda
):
    t0 = time.perf_counter()
    retr = WarmRetriever(
        duckdb_path=duckdb_path,
        fts_path=fts_path,
        hnsw_path=hnsw_path,
        ids_npy=ids_npy,
        embs_npy=embs_npy,
        encoder_name=model_name,
        k_lex=k_lex, k_ann=k_ann, fuse_k=fuse_k, k_final=k_final, mmr_lambda=mmr_lambda,
    )
    startup_ms = (time.perf_counter() - t0) * 1000
    return retr, round(startup_ms, 1)

retr, startup_ms = load_retriever(
    duckdb_path, fts_path, hnsw_path, ids_npy, embs_npy, model_name,
    k_lex, k_ann, fuse_k, k_final, mmr_lambda
)

# -----------
# Main pane
# -----------
st.title("🔎 Hyper Librarian — 10-K Q&A")
st.caption(f"Warm startup: {startup_ms} ms. Query time should be ~tens of ms on CPU.")

q = st.text_input("Ask a question about the filings", placeholder='e.g., "FX risk in 2023" or "What drove revenue growth in 2023?"')
go = st.button("Answer", type="primary", use_container_width=False)

if go and q.strip():
    with st.spinner("Retrieving…"):
        chunk_rows, timings = retr.retrieve(q.strip())

    with st.spinner("Composing answer…"):
        picks = select_sentences(retr.duck, chunk_rows, q.strip(), target=target_sents)
        ok = answerability(picks, min_sources=min_sources)
        answer_text = compose_answer(picks if ok else [])

    # Answer card
    colA, colB = st.columns([2,1], gap="large")

    with colA:
        st.subheader("Answer")
        status = "✅ answerable" if ok else "⚠️ low support"
        st.markdown(f"*{status}*")
        st.write(answer_text or "No answer produced.")

        st.markdown("**Citations**")
        if not picks:
            st.caption("No citations.")
        else:
            for i, p in enumerate(picks, start=1):
                st.markdown(
                    f"{i}. `{p['ticker']} {p['year']} • {p['item']} • sent {p['sent_idx']} • {p['hash']}` "
                    f"(doc {p['doc_id']}, section {p['section_id']}, sent_id {p['sent_id']})"
                )

    with colB:
        st.subheader("Timings (ms)")
        if timings:
            for k, v in timings.items():
                st.write(f"{k}: {round(v,2)}")
        else:
            st.caption("No timings reported")

        st.markdown("---")
        st.subheader("Evidence Chunks")
        if not chunk_rows:
            st.caption("No chunks.")
        else:
            for (chunk_id, ticker, filing_date, item, text, s0, s1, section_id, doc_id) in chunk_rows:
                preview = " ".join((text or "").split())[:320]
                with st.expander(f"{ticker} {str(filing_date)[:10]} • {item} • chunk {chunk_id} • doc {doc_id}"):
                    st.write(preview)

else:
    st.info('Try asking: **What drove revenue growth in 2023?**', icon="💡")

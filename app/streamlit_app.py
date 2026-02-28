import streamlit as st

from ingest import run_ingest
from rag import answer

st.set_page_config(page_title="Carbon Credit Assistant", page_icon="🌿", layout="centered")

st.title("Carbon Credit Assistant")
st.caption("Runs fully offline with Ollama + local embeddings + local vector DB (Chroma).")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-k chunks", min_value=2, max_value=8, value=4, step=1)
    show_debug = st.checkbox("Show retrieved chunks (debug)", value=False)

    st.divider()
    st.subheader("Index")
    if st.button("Reindex documents", use_container_width=True):
        with st.spinner("Reindexing..."):
            run_ingest(reset_index=True)
        st.success("Index rebuilt!")

st.write("Ask questions about carbon credits, MRV, additionality, baseline, leakage, permanence, etc.")

q = st.text_input("Your question")

if q:
    with st.spinner("Thinking..."):
        out = answer(q, k=k)

    # Always show the answer (even if retrieval is weak)
    st.markdown(out["answer"])

    # Retrieval diagnostics (optional)
    if out.get("retrieval_strength"):
        st.caption(
            f"Retrieval strength: {out['retrieval_strength']} "
            f"(confidence ~ {out.get('confidence', 0):.2f})"
        )

    if out.get("distances"):
        st.caption("Top distances: " + ", ".join(f"{d:.3f}" for d in out["distances"][:5]))

    if out.get("sources"):
        st.caption("Sources (retrieval): " + ", ".join(out["sources"]))

    if show_debug and out.get("retrieved_chunks"):
        st.divider()
        st.subheader("Retrieved Chunks (Debug)")
        for i, ch in enumerate(out["retrieved_chunks"], start=1):
            st.markdown(f"**{i}. {ch['source']}**")
            st.code(ch["text"][:1200])
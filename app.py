import streamlit as st
from rag import load_components, answer

st.set_page_config(page_title="VNAV Study Assistant", page_icon="🤖")
st.title("🤖 VNAV Study Assistant")
st.caption("Ask questions about MIT 16.485: Visual Navigation for Autonomous Vehicles. "
           "Answers come only from the lecture notes, with sources.")

@st.cache_resource
def get_components():
    return load_components()

db, reranker, client = get_components()

if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['file']}, page {s['page']}**")
                    st.text(s["text"])

question = st.chat_input("Ask a question about the lectures...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the notes..."):
            reply, results = answer(db, reranker, client, question)
        st.markdown(reply)
        sources = [
            {
                "file": r.metadata["source"],
                "page": r.metadata.get("page", 0) + 1,
                "text": r.page_content[:400],
            }
            for r in results
        ]
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**{s['file']}, page {s['page']}**")
                st.text(s["text"])

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
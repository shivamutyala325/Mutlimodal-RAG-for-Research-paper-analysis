import os
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")

st.title("📚 Paper Knowledge Base")
st.caption("Multimodal RAG — derived knowledge from documents")

st.divider()

try:
    resp = requests.get(f"{API}/papers", timeout=5)
    resp.raise_for_status()
    data   = resp.json()
    papers = data["papers"]
    total  = data["total"]

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Documents indexed", total)

    if total == 0:
        st.info("No papers ingested yet. Go to **Upload** in the sidebar to add your first document.")
    else:
        st.divider()
        for paper in papers:
            pid   = paper["paper_id"]
            title = paper.get("filename") or f"{pid[:8]}…"

            c1, c2, c3 = st.columns([4, 2, 1])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(pid)
            with c2:
                st.caption("✅ PDF" if paper["has_original"]  else "⬜ PDF")
                st.caption("✅ Markdown" if paper["has_markdown"] else "⬜ Markdown")
            with c3:
                if st.button("Chat →", key=f"chat_{pid}"):
                    st.session_state["selected_paper_id"] = pid
                    st.switch_page("pages/2_Chat.py")
            st.divider()

except requests.exceptions.ConnectionError:
    st.error(f"Cannot reach backend at **{API}**. Start the server with `cd backend && python server.py`")
except Exception as e:
    st.error(f"Error: {e}")

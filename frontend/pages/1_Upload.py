import os
import time
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Upload Paper", page_icon="📤")
st.title("📤 Upload a Research Paper")
st.caption("PDF is parsed, chunked, embedded, and stored in the knowledge base.")

st.divider()

uploaded = st.file_uploader("Drop a PDF here", type=["pdf"])

if uploaded:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Ingest into Knowledge Base", type="primary"):
        with st.spinner("Uploading…"):
            try:
                resp = requests.post(
                    f"{API}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()

                st.session_state["ingest_paper_id"] = result["paper_id"]
                st.session_state["ingest_status"]   = result["status"]
                st.session_state["ingest_message"]  = result["message"]

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {API}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ── Status display ────────────────────────────────────────────────────────────

if "ingest_paper_id" in st.session_state:
    paper_id = st.session_state["ingest_paper_id"]
    status   = st.session_state.get("ingest_status", "")
    message  = st.session_state.get("ingest_message", "")

    st.divider()
    st.subheader("Ingestion Status")
    st.code(f"paper_id: {paper_id}")

    if status == "already_indexed":
        st.success(f"✅ {message}")
        st.caption("This document is already in the knowledge base — no re-processing needed.")

    elif status == "completed":
        st.success("✅ Ingestion complete. Paper is ready in the knowledge base.")
        if st.button("Go to Chat →"):
            st.session_state["selected_paper_id"] = paper_id
            st.switch_page("pages/2_Chat.py")

    elif status == "failed":
        st.error(f"❌ Ingestion failed: {message}")

    elif status in ("queued", "processing"):
        label = "⏳ Queued — waiting to start" if status == "queued" else "⚙️ Processing — pipeline running"
        with st.status(label, expanded=True):
            st.write(message)
            st.write("This may take a few minutes (vision LLM + embedding).")

        time.sleep(4)
        try:
            poll = requests.get(f"{API}/ingest/{paper_id}/status", timeout=5)
            poll.raise_for_status()
            new = poll.json()
            st.session_state["ingest_status"]  = new["status"]
            st.session_state["ingest_message"] = new["message"]
        except Exception:
            pass
        st.rerun()

import os
import uuid
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
st.title("💬 Chat with the Knowledge Base")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]

    session_id = st.text_input("Session ID", value=st.session_state["session_id"])
    st.session_state["session_id"] = session_id

    if st.button("New Session", use_container_width=True):
        st.session_state["session_id"]   = str(uuid.uuid4())[:8]
        st.session_state["chat_history"] = []
        st.session_state.pop("selected_paper_id", None)
        st.rerun()

    st.divider()
    st.header("Retrieval Scope")
    st.caption("Restrict retrieval to one document, or search all.")

    paper_options = {"🌐  All papers": None}
    try:
        resp = requests.get(f"{API}/papers", timeout=5)
        for p in resp.json().get("papers", []):
            pid = p["paper_id"]
            label = p.get("filename") or f"{pid[:8]}…"
            paper_options[f"📄  {label}"] = pid
    except Exception:
        pass

    scope_label = st.selectbox("Scope", list(paper_options.keys()))
    scope_paper_id = paper_options[scope_label]

    # Override with library navigation if set
    if st.session_state.get("selected_paper_id"):
        scope_paper_id = st.session_state["selected_paper_id"]

    st.divider()
    if st.button("Clear History", use_container_width=True):
        try:
            requests.delete(f"{API}/chat/{session_id}", timeout=5)
        except Exception:
            pass
        st.session_state["chat_history"] = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def render_context(context: list):
    if not context:
        return
    with st.expander(f"📎 {len(context)} retrieved chunk(s)", expanded=False):
        for i, chunk in enumerate(context):
            ctype = chunk["chunk_type"]
            st.caption(f"Chunk {i + 1} — {ctype.upper()}")

            if ctype == "image":
                image_path = chunk["metadata"].get("image_path", "")
                caption    = chunk["metadata"].get("caption", "")
                if image_path:
                    parts    = image_path.split("/")
                    pid_part = parts[0]
                    filename = parts[-1]
                    try:
                        url_resp = requests.get(f"{API}/papers/{pid_part}/images/{filename}", timeout=5)
                        if url_resp.ok:
                            st.image(url_resp.json()["url"], caption=caption or filename, use_container_width=True)
                    except Exception:
                        pass
                st.markdown(f"*{chunk['content'][:300]}*")

            elif ctype == "table":
                raw = chunk["content"]
                if "Raw Table:" in raw:
                    raw = raw.split("Raw Table:")[-1].strip()
                st.markdown(raw[:600])

            else:
                st.markdown(chunk["content"][:400])

            if i < len(context) - 1:
                st.divider()

# Render existing messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_context(msg.get("context", []))

# ── Input ─────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask anything about the papers…"):
    st.session_state["chat_history"].append({"role": "user", "content": prompt, "context": []})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API}/chat/{session_id}",
                    json={"message": prompt, "paper_id": scope_paper_id},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()

                response_text = result["response"]
                context       = result.get("context", [])

                st.markdown(response_text)
                render_context(context)

                st.session_state["chat_history"].append({
                    "role":    "assistant",
                    "content": response_text,
                    "context": context,
                })

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {API}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The LLM may still be processing — try again.")
            except Exception as e:
                st.error(f"Error: {e}")

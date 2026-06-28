import streamlit as st
import os
import uuid
import json
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

from self_rag import app as rag_app, set_retriever

# ── Retriever Persistence ─────────────────────────────
RETRIEVERS_DIR = Path("retrievers")
RETRIEVERS_DIR.mkdir(exist_ok=True)

def save_retriever(user_id: str, retriever):
    save_path = RETRIEVERS_DIR / user_id
    retriever.vectorstore.save_local(str(save_path))

def load_retriever(user_id: str):
    save_path = RETRIEVERS_DIR / user_id
    if not save_path.exists():
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = FAISS.load_local(
            str(save_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vs.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        print(f"❌ Retriever load failed: {e}")
        return None

# ── Session Storage ───────────────────────────────────
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def load_session(user_id: str) -> dict:
    f = SESSIONS_DIR / f"{user_id}.json"
    if f.exists():
        return json.loads(f.read_text())
    return {
        "chats": {"Chat 1": []},
        "active_chat": "Chat 1",
        "total_queries": 0,
        "successful_queries": 0,
        "pdf_names": [],
    }

def save_session(user_id: str, data: dict):
    f = SESSIONS_DIR / f"{user_id}.json"
    f.write_text(json.dumps(data, indent=2))

# ── Build Retriever ───────────────────────────────────
def build_retriever(uploaded_files):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        all_docs.extend(splitter.split_documents(docs))
        os.unlink(tmp_path)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(all_docs, embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Self-RAG Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #cba6f7; }
    .metric-label { font-size: 0.85rem; color: #a6adc8; }
    .status-fully   { color: #a6e3a1; font-weight: bold; }
    .status-partial { color: #f9e2af; font-weight: bold; }
    .status-no      { color: #f38ba8; font-weight: bold; }
    .status-useful  { color: #a6e3a1; font-weight: bold; }
    .status-not     { color: #f38ba8; font-weight: bold; }
    .source-chip {
        background: #313244; border-radius: 20px;
        padding: 4px 12px; font-size: 0.8rem;
        color: #cdd6f4; display: inline-block; margin: 2px;
    }
    .chat-user {
        background: #313244; border-radius: 10px;
        padding: 10px 15px; margin: 5px 0; color: #cdd6f4;
    }
    .chat-bot {
        background: #1e1e2e; border-left: 3px solid #cba6f7;
        border-radius: 10px; padding: 10px 15px;
        margin: 5px 0; color: #cdd6f4;
    }
</style>
""", unsafe_allow_html=True)

# ── User ID ───────────────────────────────────────────
if "user_id" not in st.query_params:
    st.query_params["user_id"] = str(uuid.uuid4())[:8]

user_id = st.query_params["user_id"]

# ── Load Session ──────────────────────────────────────
if "loaded" not in st.session_state:
    data = load_session(user_id)
    st.session_state.chats               = data["chats"]
    st.session_state.active_chat         = data["active_chat"]
    st.session_state.total_queries       = data["total_queries"]
    st.session_state.successful_queries  = data["successful_queries"]
    st.session_state.pdf_names           = data.get("pdf_names", [])
    st.session_state.loaded              = True
    st.session_state.retriever           = None

    saved_retriever = load_retriever(user_id)
    if saved_retriever:
        st.session_state.retriever = saved_retriever
        set_retriever(saved_retriever)
        print(f"✅ Retriever restored for {user_id}")

if "pdf_names"   not in st.session_state: st.session_state.pdf_names   = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "retriever"   not in st.session_state: st.session_state.retriever   = None

# Har page load pe retriever restore karo
if "retriever" in st.session_state and st.session_state.retriever is not None:
    set_retriever(st.session_state.retriever)

# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 🔑 Session: `{user_id}`")
    st.divider()

    # ── PDF Upload ────────────────────────────────
    st.markdown("### 📄 Knowledge Base")
    uploaded = st.file_uploader(
        "PDFs upload",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded:
        if st.button("🔧 Build Knowledge Base", use_container_width=True):
            with st.spinner("PDFs index ..."):
                new_retriever = build_retriever(uploaded)
                set_retriever(new_retriever)
                st.session_state.retriever = new_retriever
                st.session_state.pdf_names = [f.name for f in uploaded]
                save_retriever(user_id, new_retriever)
                save_session(user_id, {
                    "chats": st.session_state.chats,
                    "active_chat": st.session_state.active_chat,
                    "total_queries": st.session_state.total_queries,
                    "successful_queries": st.session_state.successful_queries,
                    "pdf_names": st.session_state.pdf_names
                })
            st.success(f"✅ {len(uploaded)} PDF(s) ready!")

    if st.session_state.retriever is not None:
        set_retriever(st.session_state.retriever)

    if st.session_state.pdf_names:
        for name in st.session_state.pdf_names:
            st.markdown(f'<span class="source-chip">📄 {name}</span>',
                        unsafe_allow_html=True)

    st.divider()

    # ── Chat List ─────────────────────────────────
    st.markdown("### 💬 Chats")

    for chat_id in list(st.session_state.chats.keys()):
        is_active = chat_id == st.session_state.active_chat
        label = f"▶ {chat_id}" if is_active else chat_id
        if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.active_chat = chat_id
            save_session(user_id, {
                "chats": st.session_state.chats,
                "active_chat": chat_id,
                "total_queries": st.session_state.total_queries,
                "successful_queries": st.session_state.successful_queries,
                "pdf_names": st.session_state.pdf_names,
            })
            st.rerun()

    if st.button("➕ New Chat", use_container_width=True):
        new_id = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_id] = []
        st.session_state.active_chat = new_id
        st.session_state.last_result = None
        save_session(user_id, {
            "chats": st.session_state.chats,
            "active_chat": new_id,
            "total_queries": st.session_state.total_queries,
            "successful_queries": st.session_state.successful_queries,
            "pdf_names": st.session_state.pdf_names,
        })
        st.rerun()

    st.divider()

    st.markdown("### 🔗 Session Link")
    session_url = f"http://localhost:8501/?user_id={user_id}"
    st.code(session_url)
    st.caption("Save karo — wapas aane pe same chat milega!")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chats[st.session_state.active_chat] = []
        st.session_state.last_result = None
        save_session(user_id, {
            "chats": st.session_state.chats,
            "active_chat": st.session_state.active_chat,
            "total_queries": st.session_state.total_queries,
            "successful_queries": st.session_state.successful_queries,
            "pdf_names": st.session_state.pdf_names,
        })
        st.rerun()

# ════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════
st.markdown("# 🧠 Self-RAG Dashboard")
st.markdown(f"*LangGraph + Groq + FAISS* &nbsp;|&nbsp; 📂 **{st.session_state.active_chat}**")
st.divider()

# ── Metrics ───────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(st.session_state.pdf_names)}</div>
        <div class="metric-label">📄 PDFs Loaded</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{st.session_state.total_queries}</div>
        <div class="metric-label">💬 Total Queries</div>
    </div>""", unsafe_allow_html=True)

with m3:
    rate = (
        int(st.session_state.successful_queries / st.session_state.total_queries * 100)
        if st.session_state.total_queries > 0 else 0
    )
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{rate}%</div>
        <div class="metric-label">✅ Success Rate</div>
    </div>""", unsafe_allow_html=True)

with m4:
    last_issup = (
        st.session_state.last_result.get("issup", "-")
        if st.session_state.last_result else "-"
    )
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1rem">{last_issup}</div>
        <div class="metric-label">🔍 Last IsSUP</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Chat + Stats ──────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.markdown(f"### 💬 {st.session_state.active_chat}")

    messages = st.session_state.chats.get(st.session_state.active_chat, [])

    chat_container = st.container(height=450)
    with chat_container:
        if not messages:
            st.markdown("*Query poocho — PDF upload karo!*")
        for msg in messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🧠 {msg["content"]}</div>',
                            unsafe_allow_html=True)

    query = st.chat_input("ask anything...")

if query:
    messages = st.session_state.chats.get(st.session_state.active_chat, [])
    messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1

    recent_messages = messages[-4:-1] if len(messages) > 3 else messages[:-1]

    chat_history = "\n".join([
        f"{m['role'].upper()}: {m['content'][:150]}"
        for m in recent_messages
    ])

    with st.spinner("🧠 Self-RAG soch raha hai..."):
        # chat_history mat bhejo — LangGraph MongoDB se khud load karega
        initial_state = {
            "question": question_with_history,  # ← sirf current question
            "retrieval_query": query,
            "rewrite_tries": 0,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
            "retries": 0,
            "isuse": "not_useful",
            "use_reason": "",
        }

        result = rag_app.invoke(
            initial_state,
            config={
                "recursion_limit": 80,
                "configurable": {
                    "thread_id": f"{user_id}_{st.session_state.active_chat}"
                    # ↑ same user + same chat = same MongoDB thread = history restore!
                }
            }
        )

        answer = result.get("answer", "No answer found.")
        messages.append({"role": "assistant", "content": answer})
        st.session_state.chats[st.session_state.active_chat] = messages
        st.session_state.last_result = result

        if result.get("isuse") == "useful":
            st.session_state.successful_queries += 1

        save_session(user_id, {
            "chats": st.session_state.chats,
            "active_chat": st.session_state.active_chat,
            "total_queries": st.session_state.total_queries,
            "successful_queries": st.session_state.successful_queries,
            "pdf_names": st.session_state.pdf_names,
        })
        st.rerun()

# ── RIGHT: Stats ──────────────────────────────────────
with col_info:
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("### 📊 Last Query Stats")

        need = r.get("need_retrieval", False)
        st.markdown(f"**Retrieval:** {'✅ Yes' if need else '⚡ Direct'}")
        st.markdown(f"**Docs Retrieved:** {len(r.get('docs', []) or [])}")
        st.markdown(f"**Relevant Docs:** {len(r.get('relevant_docs', []) or [])}")
        st.markdown(f"**Rewrite Tries:** {r.get('rewrite_tries', 0)}")
        st.markdown(f"**Revise Tries:** {r.get('retries', 0)}")

        st.divider()

        issup = r.get("issup", "-")
        if issup == "fully_supported":
            st.markdown('**IsSUP:** <span class="status-fully">✅ Fully Supported</span>',
                        unsafe_allow_html=True)
        elif issup == "partially_supported":
            st.markdown('**IsSUP:** <span class="status-partial">⚠️ Partially</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('**IsSUP:** <span class="status-no">❌ No Support</span>',
                        unsafe_allow_html=True)

        evidence = r.get("evidence", []) or []
        if evidence:
            with st.expander("🔎 Evidence"):
                for e in evidence:
                    st.markdown(f"- *{e}*")

        st.divider()

        isuse = r.get("isuse", "-")
        if isuse == "useful":
            st.markdown('**IsUSE:** <span class="status-useful">✅ Useful</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('**IsUSE:** <span class="status-not">❌ Not Useful</span>',
                        unsafe_allow_html=True)
        st.markdown(f"*{r.get('use_reason', '')}*")

        st.divider()

        relevant_docs = r.get("relevant_docs", []) or []
        if relevant_docs:
            st.markdown("**📚 Sources:**")
            for d in relevant_docs:
                src = (d.metadata or {}).get("source", "unknown")
                page = (d.metadata or {}).get("page", "")
                label = os.path.basename(src) + (f" p.{page}" if page != "" else "")
                st.markdown(f'<span class="source-chip">📄 {label}</span>',
                            unsafe_allow_html=True)
    else:
        st.markdown("### 📊 Stats")
        st.info("Pehla question poochho — stats yahan dikhenge!")
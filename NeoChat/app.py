import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Optional Langsmith tracing — only enabled if key is present, never crashes the app
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Next-Level Q&A Chatbot With OpenAI"

st.set_page_config(
    page_title="NeoChat — AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stChatMessage { border-radius: 14px; }
    .main-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #8b8b8b; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.2rem; }
    .stat-pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: rgba(120,120,120,0.15); font-size: 0.78rem; margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "chats" not in st.session_state:
    # Each chat: {id, title, messages: [{role, content}], created}
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {
            "title": "New chat",
            "messages": [],
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    }
    st.session_state.active_chat = first_id

if "total_tokens_est" not in st.session_state:
    st.session_state.total_tokens_est = 0


def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {
        "title": "New chat",
        "messages": [],
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.active_chat = cid


def delete_chat(cid):
    if cid in st.session_state.chats:
        del st.session_state.chats[cid]
    if not st.session_state.chats:
        new_chat()
    elif st.session_state.active_chat == cid:
        st.session_state.active_chat = next(iter(st.session_state.chats))


def est_tokens(text):
    # rough heuristic: ~4 chars per token
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Sidebar — settings + chat sessions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input("OpenAI API Key", type="password", help="Never stored, only used for this session")

    engine = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        index=0,
    )

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, knowledgeable assistant. Answer clearly and concisely.",
        height=90,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    with col_b:
        max_tokens = st.slider("Max tokens", 50, 4000, 500, 50)

    stream_response = st.toggle("Stream responses", value=True)

    st.divider()
    st.subheader("💬 Chats")

    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    for cid, chat in sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True
    ):
        c1, c2 = st.columns([5, 1])
        label = chat["title"] if chat["messages"] else "New chat"
        with c1:
            if st.button(
                ("👉 " if cid == st.session_state.active_chat else "") + label,
                key=f"select_{cid}",
                use_container_width=True,
            ):
                st.session_state.active_chat = cid
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{cid}"):
                delete_chat(cid)
                st.rerun()

    st.divider()
    active = st.session_state.chats[st.session_state.active_chat]
    if active["messages"]:
        transcript = "\n\n".join(
            f"**{m['role'].upper()}:** {m['content']}" for m in active["messages"]
        )
        st.download_button(
            "⬇️ Export chat (.md)",
            data=transcript,
            file_name=f"chat_{st.session_state.active_chat[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Export chat (.json)",
            data=json.dumps(active["messages"], indent=2),
            file_name=f"chat_{st.session_state.active_chat[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(f"Est. tokens used this session: ~{st.session_state.total_tokens_est}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">💬 NeoChat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A full conversation-aware assistant powered by OpenAI + LangChain</p>', unsafe_allow_html=True)

active = st.session_state.chats[st.session_state.active_chat]

st.markdown(
    f'<span class="stat-pill">Model: {engine}</span>'
    f'<span class="stat-pill">Temp: {temperature}</span>'
    f'<span class="stat-pill">Max tokens: {max_tokens}</span>'
    f'<span class="stat-pill">Messages: {len(active["messages"])}</span>',
    unsafe_allow_html=True,
)
st.write("")

# Render history
for msg in active["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar first.")
        st.stop()

    # Title the chat from the first message
    if not active["messages"]:
        active["title"] = user_input[:40] + ("…" if len(user_input) > 40 else "")

    active["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build LangChain message history
    lc_messages = [SystemMessage(content=system_prompt)]
    for m in active["messages"]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    try:
        llm = ChatOpenAI(
            model=engine,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=stream_response,
        )

        with st.chat_message("assistant"):
            if stream_response:
                placeholder = st.empty()
                full_response = ""
                for chunk in llm.stream(lc_messages):
                    full_response += chunk.content or ""
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                with st.spinner("Thinking..."):
                    result = llm.invoke(lc_messages)
                    full_response = result.content
                st.markdown(full_response)

        active["messages"].append({"role": "assistant", "content": full_response})
        st.session_state.total_tokens_est += est_tokens(user_input) + est_tokens(full_response)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

elif not active["messages"]:
    st.info("👋 Start the conversation below. Your chat history, exports, and multiple sessions are all handled automatically.")

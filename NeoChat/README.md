# NeoChat — Next-Level OpenAI Chatbot

Upgrade over the original single-turn Q&A script.

## New features
- **Real conversation memory** — multi-turn chat, not one-shot Q&A
- **Streaming responses** — tokens appear live, ChatGPT-style
- **Multiple chat sessions** — new chat, switch, delete, auto-titled
- **Editable system prompt** — control the assistant's persona/behaviour
- **Export chat** — download as Markdown or JSON
- **Model picker** — gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
- **Token usage estimate** shown in sidebar
- **Proper error handling** — no crash on bad key / API errors
- **Clean UI** — stat pills, dark/light aware, chat-bubble layout

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter your OpenAI API key in the sidebar (not stored anywhere, session-only).

## Bugs fixed from original
- Typos ("massistant", "repsonse", "Goe ahead", "sider bar")
- `temperature` and `max_tokens` were accepted but never actually passed to the LLM
- `openai.api_key` was set globally but unused — key is now passed directly to `ChatOpenAI`
- `LANGCHAIN_API_KEY` crashed the app if the env var was missing — now optional
- Single `st.text_input` had no history — replaced with `st.chat_input` + persistent session state

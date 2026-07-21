import html
import os

import streamlit as st
import streamlit.components.v1 as components

import db
import rag

# Initialize DB on startup (graceful if DB not configured)
db.init_db()

st.set_page_config(page_title="ML Concepts RAG", page_icon="🤖")

st.title("ML Concepts Atlas")
st.markdown(
    "Ask about a machine-learning concept. Each grounded answer includes sources "
    "and a visual concept map."
)

# ── Configuration warnings ────────────────────────────────────────────────────
missing = []
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    missing.append("**LLM**: Set `OPENAI_API_KEY` or `GROQ_API_KEY` in Streamlit secrets.")
if not os.getenv("PINECONE_API_KEY"):
    missing.append("**Vector DB**: Set `PINECONE_API_KEY` in Streamlit secrets.")

if missing:
    with st.expander("⚠️ Configuration required — click to see details", expanded=True):
        st.warning(
            "The following secrets are not configured. "
            "Go to **Streamlit Cloud → App settings → Secrets** and add them:\n\n"
            + "\n".join(f"- {m}" for m in missing)
        )
        st.code(
            "# Paste this into Streamlit Cloud Secrets (TOML format)\n"
            'GROQ_API_KEY = "your_groq_key_here"\n'
            'PINECONE_API_KEY = "your_pinecone_key_here"\n'
            '# Optional:\n'
            '# OPENAI_API_KEY = "sk-..."\n'
            '# DATABASE_URL = "postgresql://..."\n',
            language="toml",
        )
# ─────────────────────────────────────────────────────────────────────────────


def render_mermaid(diagram):
    """Render generated Mermaid code in an isolated component."""
    if not diagram:
        return

    safe_diagram = html.escape(diagram)
    components.html(
        f"""
        <div class="diagram-shell">
          <div class="diagram-label">CONCEPT MAP</div>
          <pre class="mermaid">{safe_diagram}</pre>
        </div>
        <style>
          :root {{ color-scheme: light; }}
          body {{ margin: 0; background: transparent; font-family: Inter, sans-serif; }}
          .diagram-shell {{
            border: 1px solid #cbd5e1; border-left: 5px solid #2563eb;
            border-radius: 12px; padding: 14px 18px; background: #f8fafc;
          }}
          .diagram-label {{
            color: #475569; font: 700 11px/1.2 ui-monospace, monospace;
            letter-spacing: .14em; margin-bottom: 8px;
          }}
        </style>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{
            startOnLoad: true,
            securityLevel: 'strict',
            theme: 'base',
            themeVariables: {{
              primaryColor: '#dbeafe', primaryTextColor: '#0f172a',
              primaryBorderColor: '#2563eb', lineColor: '#64748b',
              secondaryColor: '#e0f2fe', tertiaryColor: '#f8fafc',
              fontFamily: 'Inter, sans-serif'
            }}
          }});
        </script>
        """,
        height=430,
        scrolling=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_mermaid(message.get("mermaid_diagram", ""))

# Feedback handlers
def handle_feedback(interaction_id, feedback_val):
    db.log_feedback(interaction_id, feedback_val)
    st.toast("Thank you for your feedback!")

if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = rag.get_rag_response(prompt)
            
            answer = response_data["answer"]
            st.markdown(answer)
            render_mermaid(response_data.get("mermaid_diagram", ""))

            # Log interaction to DB
            interaction_id = db.log_interaction(
                prompt, 
                response_data["rewritten_query"], 
                answer, 
                response_data["response_time_ms"]
            )
            
            if response_data["contexts"]:
                with st.expander("Show Sources"):
                    for i, ctx in enumerate(response_data["contexts"]):
                        st.markdown(f"**{i+1}. [{ctx.get('title', 'Source')}]({ctx.get('url', '#')})**")
                        st.text(ctx.get("text", ""))
            
            if interaction_id:
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    st.button("👍", on_click=handle_feedback, args=(interaction_id, 1), key=f"pos_{interaction_id}")
                with col2:
                    st.button("👎", on_click=handle_feedback, args=(interaction_id, -1), key=f"neg_{interaction_id}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "mermaid_diagram": response_data.get("mermaid_diagram", ""),
    })

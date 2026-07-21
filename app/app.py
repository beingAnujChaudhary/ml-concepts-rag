import html
import os

import streamlit as st
import streamlit.components.v1 as components

import db
import rag

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Concepts Atlas · beingAnujChaudhary",
    page_icon="🤖",
    layout="centered",
)

# ── Brand CSS (portfolio-matched) ─────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap" rel="stylesheet">

    <style>
    /* ── Typography ── */
    html, body, [class*="css"], .stMarkdown, .stTextInput, button {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
    }

    /* ── Background ── */
    .stApp { background-color: #EFEAE3; }
    section[data-testid="stSidebar"] { background-color: #E4DDD5; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* ── Branded header bar ── */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    .brand-logo-area {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-dot {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, #FE320A, #FF6B35);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 4px 14px rgba(254,50,10,0.3);
        flex-shrink: 0;
    }
    .brand-name {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(26,26,26,0.5);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1.2;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        padding: 0.3rem 0.85rem;
        border: 1px solid rgba(254,50,10,0.25);
        border-radius: 50px;
        color: #FE320A;
        background: rgba(254,50,10,0.06);
    }

    /* ── Hero section ── */
    .hero-section {
        text-align: center;
        padding: 2.5rem 0 2rem 0;
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #FE320A;
        border: 1px solid rgba(254,50,10,0.3);
        border-radius: 50px;
        padding: 0.3rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-heading {
        font-size: clamp(1.9rem, 4vw, 2.6rem);
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0.5rem 0;
    }
    .hero-heading span {
        background: linear-gradient(to right, #FE320A, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 1rem;
        color: rgba(26,26,26,0.55);
        font-weight: 400;
        max-width: 460px;
        margin: 0.6rem auto 0;
        line-height: 1.6;
    }

    /* ── Chat input ── */
    .stChatInput > div {
        border: 1px solid rgba(0,0,0,0.12) !important;
        border-radius: 50px !important;
        background: #fff !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .stChatInput > div:focus-within {
        border-color: #FE320A !important;
        box-shadow: 0 0 0 3px rgba(254,50,10,0.12) !important;
    }
    .stChatInput textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: #1a1a1a !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    /* User bubble */
    [data-testid="stChatMessage"][data-role="user"] .stMarkdown {
        background: #1a1a1a;
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem;
        display: inline-block;
        max-width: 80%;
        float: right;
        font-size: 0.95rem;
        line-height: 1.55;
    }
    /* Assistant bubble */
    [data-testid="stChatMessage"][data-role="assistant"] .stMarkdown {
        background: #fff;
        color: #1a1a1a;
        border-radius: 4px 18px 18px 18px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ── Avatar icons ── */
    [data-testid="stChatMessageAvatarUser"] {
        background: #1a1a1a !important;
        color: #fff !important;
        border-radius: 50% !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #FE320A, #FF6B35) !important;
        border-radius: 50% !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 50px !important;
        border: 1px solid rgba(0,0,0,0.12) !important;
        background: transparent !important;
        color: #1a1a1a !important;
        transition: background 0.25s ease, color 0.25s ease, border-color 0.25s ease,
                    box-shadow 0.25s ease !important;
        padding: 0.35rem 1.1rem !important;
    }
    .stButton > button:hover {
        background: #1a1a1a !important;
        color: #fff !important;
        border-color: #1a1a1a !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #FE320A !important; }

    /* ── Expanders (Sources) ── */
    .stExpander {
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 12px !important;
        background: #fff !important;
    }
    .stExpander summary {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: rgba(26,26,26,0.6) !important;
    }
    .stExpander summary:hover { color: #FE320A !important; }

    /* ── Config warning block ── */
    .stAlert {
        border-radius: 12px !important;
        border-left-color: #FE320A !important;
        font-size: 0.9rem !important;
    }

    /* ── Code block ── */
    .stCodeBlock { border-radius: 10px !important; }

    /* ── Divider ── */
    hr { border-color: rgba(0,0,0,0.07) !important; }

    /* ── Toast ── */
    [data-testid="stToast"] {
        background: #1a1a1a !important;
        color: #fff !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Focus ring ── */
    :focus-visible { outline: 2px solid #FE320A !important; outline-offset: 2px; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(254,50,10,0.5); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── DB init ───────────────────────────────────────────────────────────────────
db.init_db()

# ── Branded header ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-header">
        <div class="brand-logo-area">
            <div class="brand-dot">🤖</div>
            <div>
                <div class="brand-name">beingAnujChaudhary</div>
                <div class="brand-title">ML Concepts Atlas</div>
            </div>
        </div>
        <span class="brand-badge">RAG · Powered</span>
    </div>
    """,
    unsafe_allow_html=True,
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

# ── Hero ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state or not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-tag">Retrieval-Augmented Generation</div>
            <h1 class="hero-heading">Ask about <span>Machine Learning</span></h1>
            <p class="hero-sub">
                Each grounded answer pulls from a curated knowledge base
                and includes sources and a visual concept map.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def render_mermaid(diagram):
    """Render Mermaid diagram in an isolated sandboxed iframe."""
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
          body {{ margin: 0; background: transparent; font-family: 'DM Sans', sans-serif; }}
          .diagram-shell {{
            border: 1px solid rgba(0,0,0,0.08);
            border-left: 4px solid #FE320A;
            border-radius: 12px;
            padding: 14px 18px;
            background: #fff;
          }}
          .diagram-label {{
            color: rgba(26,26,26,0.4);
            font: 600 10px/1.2 ui-monospace, monospace;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: 10px;
          }}
        </style>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{
            startOnLoad: true,
            securityLevel: 'strict',
            theme: 'base',
            themeVariables: {{
              primaryColor: '#fff0ec',
              primaryTextColor: '#1a1a1a',
              primaryBorderColor: '#FE320A',
              lineColor: '#888',
              secondaryColor: '#EFEAE3',
              tertiaryColor: '#f9f7f4',
              fontFamily: "'DM Sans', sans-serif"
            }}
          }});
        </script>
        """,
        height=430,
        scrolling=True,
    )


def handle_feedback(interaction_id, feedback_val):
    db.log_feedback(interaction_id, feedback_val)
    st.toast("Thanks for the feedback!")


# ── Session history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_mermaid(message.get("mermaid_diagram", ""))

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Tell me about machine learning…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response_data = rag.get_rag_response(prompt)

        answer = response_data["answer"]
        st.markdown(answer)
        render_mermaid(response_data.get("mermaid_diagram", ""))

        interaction_id = db.log_interaction(
            prompt,
            response_data["rewritten_query"],
            answer,
            response_data["response_time_ms"],
        )

        if response_data["contexts"]:
            with st.expander("📎 Sources"):
                for i, ctx in enumerate(response_data["contexts"]):
                    title = ctx.get("title", f"Source {i+1}")
                    url = ctx.get("url", "#")
                    text = ctx.get("text", "")
                    st.markdown(f"**{i+1}. [{title}]({url})**")
                    st.caption(text[:300] + ("…" if len(text) > 300 else ""))

        if interaction_id:
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                st.button(
                    "👍", key=f"pos_{interaction_id}",
                    on_click=handle_feedback, args=(interaction_id, 1),
                )
            with col2:
                st.button(
                    "👎", key=f"neg_{interaction_id}",
                    on_click=handle_feedback, args=(interaction_id, -1),
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "mermaid_diagram": response_data.get("mermaid_diagram", ""),
        }
    )

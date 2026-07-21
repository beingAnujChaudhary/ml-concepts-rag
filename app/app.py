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
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }
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
    /* ── Sidebar Profile ── */
    .sidebar-profile {
        text-align: center;
        padding: 1rem 0;
    }
    .profile-name {
        font-weight: 600;
        font-size: 1.25rem;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .profile-handle {
        font-size: 0.85rem;
        color: rgba(26,26,26,0.6);
        margin-bottom: 1.5rem;
    }
    .social-links {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        justify-content: center;
    }
    .social-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(26,26,26,0.05);
        color: #1a1a1a;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    .social-link:hover {
        background: #FE320A;
        color: #fff;
        transform: translateY(-2px);
    }
    .social-link svg {
        width: 18px;
        height: 18px;
        fill: currentColor;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── DB init ───────────────────────────────────────────────────────────────────
db.init_db()

# ── Sidebar Profile ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-profile">
            <div class="profile-name">Anuj Chaudhary</div>
            <div class="profile-handle">@beinganujchaudhary</div>
            
            <div class="social-links">
                <a href="https://www.linkedin.com/in/beinganujchaudhary/" target="_blank" class="social-link" title="LinkedIn">
                    <svg viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                </a>
                <a href="https://github.com/beinganujchaudhary" target="_blank" class="social-link" title="GitHub">
                    <svg viewBox="0 0 24 24"><path fill-rule="evenodd" d="M12 0C5.373 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.565 21.795 24 17.3 24 12c0-6.627-5.373-12-12-12z" clip-rule="evenodd"/></svg>
                </a>
                <a href="https://twitter.com/anujisonholiday" target="_blank" class="social-link" title="X (Twitter)">
                    <svg viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                </a>
                <a href="https://www.instagram.com/beinganujchaudhary" target="_blank" class="social-link" title="Instagram">
                    <svg viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919C8.416 2.175 8.796 2.163 12 2.163m0-2.163C8.74.001 8.344.001 7.053.058c-3.75.17-6.175 1.649-6.98 6.98-.056 1.291-.058 1.687-.058 4.914s.002 3.623.058 4.914c.805 5.332 3.23 6.811 6.98 6.98 1.291.056 1.687.058 4.914.058s3.623-.002 4.914-.058c5.332-.805 6.811-3.23 6.98-6.98.056-1.291.058-1.687.058-4.914s-.002-3.623-.058-4.914c-.805-5.332-3.23-6.811-6.98-6.98C15.656.001 15.26.001 12 .001z"></path></svg>
                </a>
                <a href="mailto:beinganujchaudhary@gmail.com" class="social-link" title="Email">
                    <svg viewBox="0 0 24 24"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
                </a>
                <a href="https://wa.me/message/QT5MJZ25KRYDA1" target="_blank" class="social-link" title="WhatsApp">
                    <svg viewBox="0 0 32 32"><path d="M16 2C8.28 2 2 8.28 2 16c0 2.46.66 4.88 1.9 7.02L2 30l7.18-1.88A13.94 13.94 0 0016 30c7.72 0 14-6.28 14-14S23.72 2 16 2zm0 25.5c-2.18 0-4.32-.58-6.2-1.68l-.44-.26-4.26 1.12 1.14-4.14-.28-.46A11.44 11.44 0 014.5 16C4.5 9.6 9.6 4.5 16 4.5S27.5 9.6 27.5 16 22.4 27.5 16 27.5zm6.28-8.56c-.34-.17-2.02-1-2.34-1.11-.32-.11-.55-.17-.78.17-.23.34-.9 1.11-1.1 1.34-.2.23-.4.26-.74.09-.34-.17-1.44-.53-2.74-1.69-1.01-.9-1.7-2.01-1.9-2.35-.2-.34-.02-.52.15-.69.15-.15.34-.4.51-.6.17-.2.23-.34.34-.57.11-.23.06-.43-.03-.6-.09-.17-.78-1.88-1.07-2.57-.28-.68-.57-.59-.78-.6h-.66c-.23 0-.6.09-.91.43-.31.34-1.19 1.16-1.19 2.83s1.22 3.28 1.39 3.51c.17.23 2.4 3.66 5.82 5.13.81.35 1.45.56 1.94.72.82.26 1.56.22 2.15.13.66-.1 2.02-.82 2.31-1.62.29-.8.29-1.48.2-1.62-.09-.14-.32-.23-.66-.4z"/></svg>
                </a>
            </div>
            
            <div style="margin-top: 2.5rem; font-size: 0.82rem; color: rgba(26,26,26,0.55); line-height: 1.5;">
                <p>Welcome to the <strong>ML Concepts Atlas</strong>. This RAG-powered bot uses a curated knowledge base to teach Machine Learning with context and visual concept maps.</p>
                <p style="margin-top: 1rem;">&copy; 2024 Anuj Chaudhary</p>
                <p style="margin-top: 0.2rem;"><a href="https://beinganujchaudhary.web.app/" target="_blank" style="color: #FE320A; text-decoration: none;">beinganujchaudhary.web.app</a></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

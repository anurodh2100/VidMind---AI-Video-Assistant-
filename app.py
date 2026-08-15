import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VidMind · AI Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

THEMES = {
    "dark": {
        "bg": "#171718",
        "bg_glow_1": "rgba(90,140,255,0.12)",
        "bg_glow_2": "rgba(140,90,255,0.08)",
        "text": "#e8e4dc",
        "text_bright": "#f0ebe0",
        "text_dim": "#a09890",
        "text_dimmer": "#605850",
        "card_bg": "rgba(255,255,255,0.03)",
        "card_border": "rgba(255,255,255,0.07)",
        "input_bg": "rgba(255,255,255,0.05)",
        "panel_bg": "rgba(255,255,255,0.025)",
        "accent": "#5a8cff",
        "accent_border": "rgba(90,140,255,0.25)",
        "accent_border_soft": "rgba(90,140,255,0.15)",
        "green": "#50c878",
        "green_border": "rgba(80,200,120,0.2)",
        "green_border_soft": "rgba(80,200,120,0.15)",
        "purple": "#a78bfa",
    },
    "light": {
        "bg": "#faf8f5",
        "bg_glow_1": "rgba(90,140,255,0.08)",
        "bg_glow_2": "rgba(140,90,255,0.06)",
        "text": "#2a2620",
        "text_bright": "#1a1712",
        "text_dim": "#6b6258",
        "text_dimmer": "#9a9186",
        "card_bg": "rgba(0,0,0,0.02)",
        "card_border": "rgba(0,0,0,0.08)",
        "input_bg": "rgba(0,0,0,0.03)",
        "panel_bg": "rgba(0,0,0,0.015)",
        "accent": "#3162e0",
        "accent_border": "rgba(49,98,224,0.3)",
        "accent_border_soft": "rgba(49,98,224,0.2)",
        "green": "#2f9e5c",
        "green_border": "rgba(47,158,92,0.3)",
        "green_border_soft": "rgba(47,158,92,0.2)",
        "purple": "#7c5cff",
    },
}
t = THEMES[st.session_state.theme]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: {t["text"]}; }}

.stApp {{
    background: {t["bg"]};
    background-image:
        radial-gradient(ellipse 80% 50% at 15% -10%, {t["bg_glow_1"]} 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 110%, {t["bg_glow_2"]} 0%, transparent 55%);
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 3rem 4rem; max-width: 1200px; }}

.hero {{ text-align: center; padding: 1.2rem 0 2rem; }}
.hero-eyebrow {{
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase; color: {t["accent"]};
    margin-bottom: 1rem; opacity: 0.9;
}}
.hero h1 {{
    font-family: 'Syne', sans-serif; font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800; line-height: 1.0; letter-spacing: -0.03em;
    color: {t["text_bright"]}; margin: 0 0 1rem;
}}
.hero h1 span {{ color: {t["accent"]}; }}
.hero-sub {{
    font-size: 1.02rem; font-weight: 300; color: {t["text_dim"]};
    max-width: 560px; margin: 0 auto; line-height: 1.65;
}}

.divider {{
    height: 1px; margin: 1.8rem 0;
    background: linear-gradient(90deg, transparent, {t["accent_border_soft"]}, transparent);
}}

.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {{
    background: {t["input_bg"]} !important;
    border: 1px solid {t["accent_border_soft"]} !important;
    border-radius: 10px !important;
    color: {t["text_bright"]} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {t["accent"]} !important;
    box-shadow: 0 0 0 3px {t["accent_border_soft"]} !important;
}}
.stTextInput > label, .stSelectbox > label, .stRadio > label, .stFileUploader > label {{
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: {t["accent"]} !important; font-weight: 500 !important;
}}
.stTextInput input::placeholder {{ color: {t["text_dim"]} !important; opacity: 0.85 !important; }}

.stButton > button {{
    background: linear-gradient(135deg, {t["accent"]} 0%, {t["purple"]} 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 0.03em !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important; cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px {t["accent_border_soft"]} !important;
    width: 100%;
}}
.stButton > button:hover {{ transform: translateY(-2px) !important; opacity: 0.95 !important; }}

.step-card {{
    background: {t["card_bg"]}; border: 1px solid {t["card_border"]};
    border-radius: 14px; padding: 1rem 1.4rem; margin-bottom: 0.7rem;
    position: relative; overflow: hidden; transition: border-color 0.3s;
}}
.step-card.active {{ border-color: {t["accent_border"]}; background: {t["accent_border_soft"]}; }}
.step-card.done {{ border-color: {t["green_border"]}; background: {t["green_border_soft"]}; }}
.step-card::before {{
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    border-radius: 14px 0 0 14px; background: {t["card_border"]}; transition: background 0.3s;
}}
.step-card.active::before {{ background: {t["accent"]}; }}
.step-card.done::before   {{ background: {t["green"]}; }}
.step-row {{ display: flex; align-items: center; gap: 0.7rem; }}
.step-num {{
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.15em; color: {t["accent"]}; opacity: 0.7;
}}
.step-title {{
    font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: {t["text_bright"]};
}}
.step-status {{ margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; }}
.status-waiting {{ color: {t["text_dimmer"]}; }}
.status-running {{ color: {t["accent"]}; }}
.status-done    {{ color: {t["green"]}; }}

.result-panel {{
    background: {t["panel_bg"]}; border: 1px solid {t["card_border"]};
    border-radius: 16px; padding: 1.8rem 2.2rem; margin-top: 0.5rem;
}}
.panel-label {{
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: {t["accent"]}; margin-bottom: 1rem;
    padding-bottom: 0.6rem; border-bottom: 1px solid {t["accent_border_soft"]};
}}

.section-heading {{
    font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700;
    color: {t["text_bright"]}; margin: 1.6rem 0 0.8rem;
}}

.footer {{
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {t["text_dimmer"]};
    text-align: center; margin-top: 3rem; letter-spacing: 0.08em;
}}
.footer b {{ color: {t["accent"]}; }}

/* Force color on Streamlit's own rendered text everywhere */
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {{
    background: transparent !important; color: {t["text"]} !important;
}}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] a,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{ color: {t["text"]} !important; }}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
    color: {t["text_bright"]} !important; font-family: 'Syne', sans-serif !important;
}}
[data-testid="stAlert"] {{ background: {t["card_bg"]} !important; border: 1px solid {t["card_border"]} !important; }}
[data-testid="stAlert"] p {{ color: {t["text"]} !important; }}
[data-testid="stExpander"] {{ background: transparent !important; border: 1px solid {t["card_border"]} !important; border-radius: 10px !important; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color: {t["text"]} !important; }}
[data-testid="stChatMessage"] {{ background: {t["card_bg"]} !important; border: 1px solid {t["card_border"]} !important; border-radius: 12px !important; }}
[data-testid="stChatMessage"] p {{ color: {t["text"]} !important; }}
[data-testid="stChatInput"] textarea {{ color: {t["text_bright"]} !important; }}
[data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, {t["accent"]} 0%, {t["purple"]} 100%) !important;
    color: #0a0a0f !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
}}
.stTabs [data-baseweb="tab"] {{ color: {t["text_dim"]} !important; font-family: 'DM Mono', monospace !important; }}
.stTabs [aria-selected="true"] {{ color: {t["accent"]} !important; }}
</style>
""", unsafe_allow_html=True)


def step_card(num, title, state):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done": ("✓ DONE", "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-row">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────────
defaults = {
    "theme": "light",
    "results": {},
    "stage": 0,          # 0=idle .. 8=done
    "source": "",
    "language": "english",
    "chat_history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

PIPELINE_STEPS = [
    ("chunks", "Process Input (download/split audio)"),
    ("transcript", "Transcribe Audio"),
    ("title", "Generate Title"),
    ("summary", "Summarize Content"),
    ("action_items", "Extract Action Items"),
    ("key_decisions", "Extract Key Decisions"),
    ("open_questions", "Extract Open Questions"),
    ("rag_chain", "Build Chat Index (RAG)"),
]

busy = st.session_state.stage not in (0, 8)

# ── Top bar: theme toggle ─────────────────────────────────────────────────────
tcol1, tcol2 = st.columns([10, 1])
with tcol2:
    icon = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(icon, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI Video Intelligence</div>
    <h1>Vid<span>Mind</span></h1>
    <p class="hero-sub">
        Drop in a YouTube link or a local recording — get a transcript, summary,
        action items, key decisions, and a chat interface to ask the video anything.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Input + pipeline (always visible, no sidebar) ─────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    input_mode = st.radio("Source", ["YouTube URL", "Upload File"], disabled=busy, horizontal=True)

    source_value = ""
    if input_mode == "YouTube URL":
        source_value = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...", disabled=busy
        )
    else:
        uploaded = st.file_uploader(
            "Upload audio/video", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"], disabled=busy
        )
        if uploaded is not None:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source_value = tmp_path

    language = st.selectbox("Language", ["english", "hinglish"], disabled=busy)
    analyze_btn = st.button("🚀 Analyze Video", use_container_width=True, disabled=busy)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    for i, (key, label) in enumerate(PIPELINE_STEPS):
        state = "done" if key in st.session_state.results else (
            "running" if st.session_state.stage == i + 1 else "waiting"
        )
        step_card(f"{i+1:02d}", label, state)

# ── Trigger pipeline ──────────────────────────────────────────────────────────
if analyze_btn:
    if not source_value.strip():
        st.warning("Please provide a YouTube URL or upload a file first.")
    else:
        st.session_state.source = source_value.strip()
        st.session_state.language = language
        st.session_state.results = {}
        st.session_state.chat_history = []
        st.session_state.stage = 1
        st.rerun()

# ── Run ONE pipeline step per rerun (live progress) ───────────────────────────
stage = st.session_state.stage
res = st.session_state.results

try:
    if stage == 1:
        with st.spinner("📥  Processing input (downloading / chunking audio)…"):
            res["chunks"] = process_input(st.session_state.source)
        st.session_state.stage = 2
        st.rerun()

    elif stage == 2:
        with st.spinner("🎙️  Transcribing audio…"):
            res["transcript"] = transcribe_all(res["chunks"], language=st.session_state.language)
        st.session_state.stage = 3
        st.rerun()

    elif stage == 3:
        with st.spinner("🏷️  Generating title…"):
            res["title"] = generate_title(res["transcript"])
        st.session_state.stage = 4
        st.rerun()

    elif stage == 4:
        with st.spinner("📋  Summarizing content…"):
            res["summary"] = summarize(res["transcript"])
        st.session_state.stage = 5
        st.rerun()

    elif stage == 5:
        with st.spinner("✅  Extracting action items…"):
            res["action_items"] = extract_action_items(res["transcript"])
        st.session_state.stage = 6
        st.rerun()

    elif stage == 6:
        with st.spinner("🔑  Extracting key decisions…"):
            res["key_decisions"] = extract_key_decisions(res["transcript"])
        st.session_state.stage = 7
        st.rerun()

    elif stage == 7:
        with st.spinner("❓  Extracting open questions…"):
            res["open_questions"] = extract_questions(res["transcript"])
        st.session_state.stage = 8
        st.rerun()

    elif stage == 8 and "rag_chain" not in res:
        with st.spinner("🧠  Building chat index for Q&A…"):
            res["rag_chain"] = build_rag_chain(res["transcript"])
        st.rerun()

except Exception as e:
    st.error(f"Pipeline failed at step {stage}: {e}")
    st.session_state.stage = 0

# ── Results ───────────────────────────────────────────────────────────────────
if res.get("summary") is not None and st.session_state.stage == 8:
    st.markdown(f'<div class="panel-label" style="border:none;">📌 {res.get("title","Untitled")}</div>', unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(res["summary"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇ Download Summary", data=res["summary"],
            file_name="summary.md", mime="text/markdown"
        )

    with tab_actions:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(res["action_items"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(res["key_decisions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(res["open_questions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_transcript:
        with st.expander("Full transcript", expanded=False):
            st.markdown(f'<div class="result-panel"><div style="white-space:pre-wrap;">{res["transcript"]}</div></div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download Transcript", data=res["transcript"],
            file_name="transcript.txt", mime="text/plain"
        )

    with tab_chat:
        st.markdown('<div class="section-heading">Chat with your video</div>', unsafe_allow_html=True)

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        question = st.chat_input("Ask something about the video…")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(res["rag_chain"], question)
                    except Exception as e:
                        answer = f"⚠️ Couldn't answer that: {e}"
                st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("🔄  Analyze a New Video", use_container_width=False):
        st.session_state.results = {}
        st.session_state.stage = 0
        st.session_state.chat_history = []
        st.session_state.source = ""
        st.rerun()

elif st.session_state.stage == 0:
    st.info("👆 Add a YouTube URL or upload a file above, then click **Analyze Video** to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    VidMind · AI Video Assistant · Built by <b>AJ Developers</b>
</div>
""", unsafe_allow_html=True)
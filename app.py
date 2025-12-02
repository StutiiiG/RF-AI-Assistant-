import streamlit as st
import time
from datetime import datetime
from rag_engine import RFAssistant

# Page Configuration
st.set_page_config(
    page_title="RF Engineering AI Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --bg-black: #050505;
        --bg-panel: #111111;
        --bg-panel-soft: #18181b;
        --text-main: #f9fafb;
        --text-muted: #a1a1aa;
        /* dimmer purple */
        --accent-purple: #b794f4;
        --accent-green: #22c55e;
    }

    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-black);
        color: var(--text-main);
    }

    h1, h2, h3, h4, h5 {
        color: var(--accent-purple);
        font-weight: 600;
        letter-spacing: -0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stMarkdown, p, label, .st-cq {
        color: var(--text-main);
    }

    .stButton>button {
        background: var(--accent-purple);
        color: var(--text-main);
        border: none;
        border-radius: 12px;
        padding: 10px 26px;
        font-weight: 600;
        transition: .15s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 6px 14px rgba(0,0,0,0.6);
    }

    .stTextArea textarea {
        background: var(--bg-panel);
        color: var(--text-main);
        border: 1px solid #27272a;
        border-radius: 10px;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-purple);
        box-shadow: 0 0 0 1px var(--accent-purple);
    }

    /* Hide default green success box */
    .element-container:has(.stSuccess) {
        display:none !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #09090b;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: transparent !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-purple) !important;
        font-size: 22px;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-panel-soft) !important;
        color: var(--text-main) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-panel) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header (title explicitly purple)
st.markdown("""
<div style='text-align: center; padding: 22px;
            border-radius: 12px; border:1px solid #27272a;'>
    <h1 style='color:#b794f4; margin:0;'>RF Engineering AI Assistant</h1>
    <p style='color:#e5e5e5; margin-top:8px;'>Instant answers from Apple RF patents & 5G research</p>
</div>
""", unsafe_allow_html=True)

# Initialize assistant
@st.cache_resource(show_spinner=False)
def load_assistant():
    assistant = RFAssistant(documents_folder="documents")
    assistant.load_documents()
    return assistant

with st.spinner("Loading document embeddings..."):
    assistant = load_assistant()

# Sidebar
with st.sidebar:
    st.markdown("### Quick Questions")
    examples = [
        "What causes antenna interference in 5G phones?",
        "How does beamforming improve mmWave?",
        "What are SAR compliance requirements?",
        "How do you reduce mutual coupling?"
    ]
    for i, q in enumerate(examples):
        if st.button(q, key=f"ex_{i}"):
            st.session_state.user_question = q
            st.rerun()

    st.markdown("---")
    st.metric("Docs Indexed", "5")
    st.metric("Chunks", "81")
    st.metric("Engine", "FAISS + GPT-4")

st.markdown("### Ask Your Question")

# Input
user_q = st.text_area(
    "",
    value=st.session_state.get("user_question", ""),
    height=120,
    placeholder="Example: How does beamforming improve 5G performance?"
)

colA, colB, _ = st.columns([1, 1, 4])
search = colA.button("Search")
if colB.button("Clear"):
    st.session_state.user_question = ""
    st.rerun()

if search and user_q.strip():
    st.session_state.user_question = user_q

    t0 = time.time()
    answer, sources = assistant.answer_question(user_q)
    dt = time.time() - t0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time", f"{dt:.1f}s")
    c2.metric("Sources", len(sources))
    c3.metric("Relevance", f"{sum(s['score'] for s in sources)/len(sources):.0%}")
    c4.metric("Chunks", "81")

    st.markdown("### Answer")
    st.markdown(f"""
    <div style='background: var(--bg-panel); padding:18px; border-radius:8px;
                border-left:3px solid var(--accent-purple); line-height:1.6;'>
        {answer}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Citations")
    for i, src in enumerate(sources, 1):
        with st.expander(f"📄 {src['document']} ({src['score']:.0%})", expanded=False):
            st.info(src['content'])

    # Why It Matters – original gradient restored
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 16px; text-align: center; margin-top: 40px;'>
        <h3 style='color: #ffffff; margin-bottom: 20px;'>Why This Matters</h3>
        <div style='display: flex; justify-content: space-around; margin-top: 20px; flex-wrap: wrap; gap: 16px;'>
            <div>
                <p style='font-size: 32px; font-weight: 700; color: #ffffff; margin: 0;'>~45 min</p>
                <p style='color: rgba(255,255,255,0.85); margin-top: 5px;'>Manual Document Search</p>
            </div>
            <div style='font-size: 48px; color: #ffffff;'>→</div>
            <div>
                <p style='font-size: 32px; font-weight: 700; color: #34C759; margin: 0;'>~10 sec</p>
                <p style='color: rgba(255,255,255,0.85); margin-top: 5px;'>AI-Powered Search</p>
            </div>
        </div>
        <p style='color: rgba(255,255,255,0.9); margin-top: 20px; font-size: 14px;'>
            <strong>Result:</strong> 99.9% time reduction • Instant insights • Cited sources
        </p>
    </div>
    """, unsafe_allow_html=True)

elif search:
    st.warning("Please enter a question first.")

import os
import time
from datetime import datetime

import streamlit as st

from rag_engine import RFAssistant

# --------------------------------------------------------------------- #
# OpenAI key from Streamlit secrets
# --------------------------------------------------------------------- #
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --------------------------------------------------------------------- #
# Page config & CSS
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="RF Engineering AI Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0E0E0E;
    }

    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #FFFFFF;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stTextArea textarea {
        background-color: #1A1A1A;
        border-radius: 12px;
        border: 2px solid #667eea;
        color: #FFFFFF;
        font-size: 15px;
        padding: 12px;
    }

    .stTextArea textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }

    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: #667eea;
    }

    [data-testid="stMetricLabel"] {
        color: #AAAAAA;
    }

    .streamlit-expanderHeader {
        background-color: #1A1A1A;
        border-radius: 8px;
        font-weight: 600;
        color: #FFFFFF;
    }

    .answer-box {
        background-color: #1A1A1A;
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 20px 0;
        color: #FFFFFF;
    }

    .stExpander {
        background-color: #1A1A1A;
        border-radius: 8px;
    }

    .stProgress > div > div > div {
        background-color: #667eea;
    }

    div[data-baseweb="notification"] {
        background-color: #1A1A1A;
        border-left: 4px solid #667eea;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------- #
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# --------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------- #
st.markdown(
    """
<div style='text-align: center; padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0; font-size: 42px;'>RF Engineering AI Assistant</h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 18px; margin-top: 10px;'>
        Instant answers from Apple patents and 5G research papers
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- #
# Load assistant (cached)
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_assistant() -> RFAssistant:
    assistant = RFAssistant(documents_folder="documents", use_gpt=True)
    assistant.load_documents()
    return assistant


with st.spinner(" Initializing AI Assistant..."):
    assistant = load_assistant()

# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("###  Example Questions")

    example_questions = [
        "What are common causes of antenna interference in multi-band systems?",
        "How does beamforming improve 5G performance?",
        "What are the key challenges in mmWave antenna design?",
        "How do you reduce mutual coupling in MIMO antennas?",
        "What materials are best for 5G antenna substrates?",
        "Explain phased array antenna design considerations.",
        "What are SAR compliance requirements for mobile antennas?",
    ]

    for i, question in enumerate(example_questions):
        if st.button(
            f"{question[:50]}...", key=f"example_{i}", use_container_width=True
        ):
            st.session_state.user_question = question
            st.rerun()

    st.markdown("---")

    if st.session_state.query_history:
        st.markdown("###  Recent Queries")
        for i, (q, t) in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.text(q[:100] + "..." if len(q) > 100 else q)
                st.caption(f"Asked at {t}")

    st.markdown("---")

    st.markdown(
        """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #AAAAAA; font-size: 14px; margin: 5px 0;'>
            <strong>Built by:</strong> Stuti Gaonkar
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------- #
# Main interface
# --------------------------------------------------------------------- #
st.markdown("### Ask Your Question")

user_question = st.text_area(
    "",
    value=st.session_state.get("user_question", ""),
    height=120,
    placeholder=(
        "Example: What design considerations are important for compact mobile "
        "antennas in 5G devices?"
    ),
    help="Ask any technical question about RF/antenna design.",
)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    search_button = st.button(" Get Answer", type="primary", use_container_width=True)
with col2:
    if st.button(" Clear", use_container_width=True):
        st.session_state.user_question = ""
        st.rerun()

if search_button and user_question.strip():
    st.session_state.query_history.append(
        (user_question, datetime.now().strftime("%H:%M:%S"))
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text(" Searching RF documents...")
    progress_bar.progress(25)
    time.sleep(0.2)

    start_time = time.time()

    try:
        status_text.text(" Analyzing relevant content...")
        progress_bar.progress(55)

        answer, sources = assistant.answer_question(user_question)

        status_text.text(" Generating response...")
        progress_bar.progress(80)
        time.sleep(0.2)

        search_time = time.time() - start_time

        progress_bar.progress(100)
        status_text.text(" Complete")
        time.sleep(0.2)

        progress_bar.empty()
        status_text.empty()

        # Metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Search Time", f"{search_time:.2f}s")
        with metric_col2:
            st.metric("Sources Found", len(sources))
        with metric_col3:
            avg_rel = (
                sum(s["score"] for s in sources) / max(len(sources), 1)
                if sources
                else 0
            )
            st.metric("Avg Relevance", f"{avg_rel:.0%}")
        with metric_col4:
            st.metric("Chunks Searched", str(getattr(assistant.index, "ntotal", "–")))

        st.markdown("---")

        # Answer
        st.markdown("### Answer")
        st.markdown(
            f"""
        <div class='answer-box'>
            {answer}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Sources
        st.markdown("### Sources & Citations")
        st.caption("Click to expand each source and view the original text.")

        for i, source in enumerate(sources, 1):
            score = source["score"]
            with st.expander(
                f"Source {i}: {source['document']} • Relevance: {score:.0%}",
                expanded=False,
            ):
                col_s1, col_s2 = st.columns([3, 1])
                with col_s1:
                    st.markdown(f"**Document:** `{source['document']}`")
                with col_s2:
                    st.markdown(
                        f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white; padding: 8px; border-radius: 8px;
                                text-align: center; font-weight: 600;'>
                        {score:.0%} Match
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown("**Excerpt:**")
                st.info(source["content"])

        st.markdown("---")

        # Export button
        if st.button("Export Results to Text File"):
            export_text = f"""RF ENGINEERING AI ASSISTANT - QUERY RESULTS
{'='*80}

QUESTION:
{user_question}

ANSWER:
{answer}

SOURCES:
{'='*80}
"""
            for i, source in enumerate(sources, 1):
                export_text += f"""
Source {i}: {source['document']}
Relevance: {source['score']:.0%}
Content: {source['content']}
{'-'*80}
"""

            st.download_button(
                label="Download Results",
                data=export_text,
                file_name=f"rf_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error("Error while answering your question.")
        st.code(str(e))
        st.info("Check the server logs for more detail or try a simpler question.")

elif search_button:
    st.warning("Please enter a question first!")

# --------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------- #
st.markdown("---")
st.markdown(
    """
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px; border-radius: 16px; text-align: center; margin-top: 40px;
            border: 1px solid #667eea;'>
    <h3 style='color: #FFFFFF; margin-bottom: 20px;'> Why This Matters</h3>
    <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
        <div>
            <p style='font-size: 32px; font-weight: 700; color: #e74c3c; margin: 0;'>~45 min</p>
            <p style='color: #AAAAAA; margin-top: 5px;'>Manual Document Search</p>
        </div>
        <div style='font-size: 48px; color: #667eea;'>→</div>
        <div>
            <p style='font-size: 32px; font-weight: 700; color: #2ecc71; margin: 0;'>~2 sec</p>
            <p style='color: #AAAAAA; margin-top: 5px;'>AI-Powered Search</p>
        </div>
    </div>
    <p style='color: #AAAAAA; margin-top: 20px; font-size: 14px;'>
        <strong style='color: #667eea;'>Result:</strong> 99.9% time reduction • Instant insights • Cited sources
    </p>
</div>
""",
    unsafe_allow_html=True,
)

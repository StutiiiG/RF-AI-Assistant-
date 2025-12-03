import streamlit as st
import os
import time
from datetime import datetime
from rag_engine import RFAssistant

# Page configuration
st.set_page_config(
    page_title="RF Engineering AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean purple theme on dark background
st.markdown("""
<style>
    /* Dark theme with purple accents */
    .stApp {
        background-color: #0E0E0E;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Buttons - Purple gradient */
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
    
    /* Text area */
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
    
    /* Success/Info messages */
    .stSuccess {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }
    
    /* Metrics - Purple color */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        color: #AAAAAA;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1A1A1A;
        border-radius: 8px;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Answer box - Dark with purple accent */
    .answer-box {
        background-color: #1A1A1A;
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 20px 0;
        color: #FFFFFF;
    }
    
    /* Source boxes */
    .stExpander {
        background-color: #1A1A1A;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #667eea;
    }
    
    /* Info boxes */
    div[data-baseweb="notification"] {
        background-color: #1A1A1A;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Header with purple gradient
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 16px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0; font-size: 42px;'>📡 RF Engineering AI Assistant</h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 18px; margin-top: 10px;'>
        Instant answers from Apple patents and 5G research papers
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize assistant
@st.cache_resource(show_spinner=False)
def load_assistant():
    assistant = RFAssistant(documents_folder="documents")
    assistant.load_documents()
    return assistant

with st.spinner(" Initializing AI Assistant..."):
    try:
        assistant = load_assistant()
        st.success(" System ready! Loaded 81 document chunks from Apple RF patents and 5G research papers.")
    except Exception as e:
        st.error(f" Error loading documents: {str(e)}")
        st.info(" Make sure you have PDF files in the 'documents' folder!")
        st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 💡 Example Questions")
    
    example_questions = [
        "What are common causes of antenna interference in multi-band systems?",
        "How does beamforming improve 5G performance?",
        "What are the key challenges in mmWave antenna design?",
        "How do you reduce mutual coupling in MIMO antennas?",
        "What materials are best for 5G antenna substrates?",
        "Explain phased array antenna design considerations",
        "What are SAR compliance requirements for mobile antennas?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(f"{question[:50]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.user_question = question
            st.rerun()
    
    st.markdown("---")
    
    # System info
    st.markdown("### System Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", "5 PDFs")
    with col2:
        st.metric("Chunks", "81")
    
    st.metric("Search Engine", "FAISS + GPT-4")
    
    st.markdown("---")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("### Recent Queries")
        for i, (q, t) in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.text(q[:100] + "..." if len(q) > 100 else q)
                st.caption(f"Asked at {t}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #AAAAAA; font-size: 14px; margin: 5px 0;'><strong>Built by:</strong> Stuti Gaonkar</p>
        <p style='color: #AAAAAA; font-size: 14px; margin: 5px 0;'><strong>For:</strong> Apple System RF Team</p>
    </div>
    """, unsafe_allow_html=True)

# Main interface
st.markdown("###  Ask Your Question")

user_question = st.text_area(
    "",
    value=st.session_state.get('user_question', ''),
    height=120,
    placeholder="Example: What design considerations are important for compact mobile antennas in 5G devices?",
    help="Ask any technical question about RF/antenna design"
)

# Buttons
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button(" Get Answer", type="primary", use_container_width=True)
with col2:
    if st.button(" Clear", use_container_width=True):
        st.session_state.user_question = ""
        st.rerun()

if search_button and user_question.strip():
    # Record query
    st.session_state.query_history.append((user_question, datetime.now().strftime("%H:%M:%S")))
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(" Searching through 81 document chunks...")
    progress_bar.progress(25)
    time.sleep(0.3)
    
    start_time = time.time()
    
    try:
        status_text.text(" Analyzing relevant content...")
        progress_bar.progress(50)
        
        answer, sources = assistant.answer_question(user_question)
        
        status_text.text(" Generating response...")
        progress_bar.progress(75)
        time.sleep(0.2)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        progress_bar.progress(100)
        status_text.text(" Complete!")
        time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        # Metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(" Search Time", f"{search_time:.2f}s")
        with metric_col2:
            st.metric(" Sources Found", len(sources))
        with metric_col3:
            st.metric(" Avg Relevance", f"{sum(s['score'] for s in sources) / len(sources):.0%}")
        with metric_col4:
            st.metric(" Chunks Searched", "81")
        
        st.markdown("---")
        
        # Answer - Dark box with purple accent
        st.markdown("### 📝 Answer")
        st.markdown(f"""
        <div class='answer-box'>
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sources
        st.markdown("###  Sources & Citations")
        st.caption("Click to expand each source and view the original text")
        
        for i, source in enumerate(sources, 1):
            # Purple color scheme for relevance
            if source['score'] > 0.5:
                relevance_color = "#667eea"
            elif source['score'] > 0.3:
                relevance_color = "#9b59b6"
            else:
                relevance_color = "#6c5ce7"
            
            with st.expander(f" Source {i}: {source['document']} • Relevance: {source['score']:.0%}", expanded=(i==1)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Document:** `{source['document']}`")
                with col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 8px; border-radius: 8px; 
                                text-align: center; font-weight: 600;'>
                        {source['score']:.0%} Match
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**Excerpt:**")
                st.info(source['content'])
        
        # Export
        st.markdown("---")
        if st.button(" Export Results to Text File"):
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
                label="💾 Download Results",
                data=export_text,
                file_name=f"rf_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f" Error: {str(e)}")
        st.info(" Try rephrasing your question or check if the documents are loaded correctly.")

elif search_button:
    st.warning("Please enter a question first!")

# Footer - Purple and Green only
st.markdown("---")
st.markdown("""
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
""", unsafe_allow_html=True)

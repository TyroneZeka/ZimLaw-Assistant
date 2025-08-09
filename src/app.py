import streamlit as st
from utils.rag_chain import ZimLawRAGChain
import time
from typing import Dict, Any

# Initialize the RAG chain
@st.cache_resource
def initialize_rag_chain() -> ZimLawRAGChain:
    """Initialize and cache the RAG chain"""
    return ZimLawRAGChain()

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Modern container styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Thinking process styling */
    .thinking-process {
        color: #6c757d;
        font-size: 0.9em;
        font-family: monospace;
        padding: 1rem;
        background: #f1f3f5;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Response container styling */
    .response-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Sources styling */
    .source-item {
        border-left: 3px solid #007bff;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def stream_response(text: str):
    """Simulate streaming response"""
    response = st.empty()
    displayed_text = ""
    
    for word in text.split():
        displayed_text += word + " "
        response.markdown(displayed_text + "▌")
        time.sleep(0.05)
    
    response.markdown(displayed_text)

def process_answer(result: Dict[str, Any]) -> None:
    """Process and display the answer with sources"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    # Show thinking process
    with st.expander("🤔 View thinking process", expanded=False):
        st.markdown('<div class="thinking-process">', unsafe_allow_html=True)
        st.markdown("""
        1. Analyzing question...
        2. Searching relevant legal documents...
        3. Extracting key sections...
        4. Formulating response...
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display the streamed answer
    st.markdown('<div class="response-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Answer")
    stream_response(result["answer"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sources
    with st.expander("📚 View Sources", expanded=True):
        st.markdown("### Referenced Legal Sections")
        if "sources" in result and result["sources"]:
            for source in result["sources"]:
                st.markdown(f"""
                <div class="source-item">
                    <strong>{source['act']}</strong><br>
                    Chapter: {source['chapter']}<br>
                    Section {source['section']}: {source['title']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific sources were cited for this answer.")

def main():
    # Page config
    st.set_page_config(
        page_title="ZimLaw Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main content
        st.title("⚖️ ZimLaw Assistant")
        st.markdown("""
        <p style='font-size: 1.2em; color: #666;'>
        Your AI-powered guide to Zimbabwean law
        </p>
        """, unsafe_allow_html=True)
        
        # Initialize RAG chain
        try:
            rag_chain = initialize_rag_chain()
        except Exception as e:
            st.error(f"Failed to initialize the legal assistant: {str(e)}")
            return
        
        # User input section
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        user_question = st.text_area(
            "What would you like to know about Zimbabwean law?",
            height=100,
            placeholder="Example: What are my rights if I'm arrested?"
        )
        
        if st.button("Get Answer", type="primary"):
            if not user_question:
                st.warning("Please enter a question.")
                return
            
            try:
                with st.spinner(""):
                    result = rag_chain.answer_question(user_question)
                    process_answer(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with col2:
        # Sidebar content now in second column
        st.markdown("""
        ### 🔍 Quick Links
        - [Constitution of Zimbabwe](https://www.constituteproject.org/constitution/Zimbabwe_2013.pdf)
        - [Labour Act](https://www.ilo.org/dyn/natlex/docs/ELECTRONIC/1850/76997/F1436867346/ZWE1850.pdf)
        - [Legal Resources](https://zimlii.org/)
        
        ### 📱 Get Help
        - Legal Aid Hotline: 116
        - Emergency: 112
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        ⚠️ <em>This is an AI-powered legal information tool. The information provided should not be considered as legal advice. 
        For specific legal matters, please consult with a qualified legal professional.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
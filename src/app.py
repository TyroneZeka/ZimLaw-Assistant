import streamlit as st
from utils.rag_chain import ZimLawRAGChain
from conditioned_answer_generator import ConditionedAnswerGenerator
import time
from typing import Dict, Any
from PIL import Image
import json
import re

# Page configuration
st.set_page_config(
    page_title="ZimLaw Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/TyroneZeka/ZimLaw-Assistant',
        'Report a bug': 'https://github.com/TyroneZeka/ZimLaw-Assistant/issues',
        'About': "# ZimLaw Assistant\nAI-powered legal research assistant for Zimbabwean law."
    }
)

@st.cache_resource
def initialize_system():
    """Initialize and cache the complete system"""
    rag_chain = ZimLawRAGChain(verbose=False)  # Clean output
    conditioned_generator = ConditionedAnswerGenerator(
        rag_chain.llm, 
        rag_chain, 
        verbose=False  # Clean output
    )
    return rag_chain, conditioned_generator

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content area */
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: white !important;
        border: 1px solid #e5e7eb;
        margin-right: 2rem;
        color: #1f2937 !important;
    }
    
    /* Legal answer formatting */
    .legal-answer {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #1f2937 !important;
    }
    
    .legal-answer h4 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem !important;
    }
    
    .legal-answer h4 strong {
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    .legal-answer ul {
        margin-left: 1rem;
    }
    
    .legal-answer li {
        margin-bottom: 0.5rem;
        color: #1f2937 !important;
    }
    
    .legal-answer p {
        color: #1f2937 !important;
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white !important;
        color: #1f2937 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Fix all text inputs */
    .stTextInput input {
        background: white !important;
        color: #1f2937 !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 10px !important;
    }
    
    /* Fix selectbox */
    .stSelectbox select {
        background: white !important;
        color: #1f2937 !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Quick actions */
    .quick-action {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0.5rem;
    }
    
    .quick-action:hover {
        border-color: #667eea;
        background: #f8fafc;
        transform: translateY(-2px);
    }
    
    /* Sources styling */
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .source-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .source-details {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .stSidebar {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        backdrop-filter: blur(20px);
    }
    
    /* Sidebar text color fixes */
    .css-1d391kg .stMarkdown, .stSidebar .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    .css-1d391kg .stMarkdown h2, .stSidebar .stMarkdown h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .css-1d391kg .stMarkdown h3, .stSidebar .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg .stMarkdown p, .stSidebar .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    .css-1d391kg .stMarkdown ul, .stSidebar .stMarkdown ul {
        color: #f1f5f9 !important;
    }
    
    .css-1d391kg .stMarkdown li, .stSidebar .stMarkdown li {
        color: #f1f5f9 !important;
    }
    
    /* Fix sidebar button text */
    .css-1d391kg .stButton button, .stSidebar .stButton button {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .css-1d391kg .stButton button:hover, .stSidebar .stButton button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Additional sidebar selectors for different Streamlit versions */
    .css-1aumxhk, .css-k1vhr4, section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    }
    
    .css-1aumxhk *, .css-k1vhr4 *, section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    .css-1aumxhk h1, .css-1aumxhk h2, .css-1aumxhk h3,
    .css-k1vhr4 h1, .css-k1vhr4 h2, .css-k1vhr4 h3,
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* Override any global text color issues */
    .stApp {
        color: #1f2937 !important;
    }
    
    /* Ensure main content text is visible */
    .main .block-container {
        color: #1f2937 !important;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    
    .loading-text {
        color: #667eea;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Metadata styling */
    .metadata-container {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Warning styling */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-text {
        color: #92400e !important;
        font-weight: 500;
    }
    
    /* Fix any remaining text visibility issues */
    * {
        color: inherit !important;
    }
    
    .stApp * {
        color: #1f2937 !important;
    }
    
    .stMarkdown {
        color: #1f2937 !important;
    }
    
    .stText {
        color: #1f2937 !important;
    }
    
    /* Placeholder text styling */
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

def format_legal_answer(answer_text: str) -> str:
    """Format legal answer with bold section headers and clean structure"""
    if not answer_text:
        return ""
    
    # Clean up any unwanted prefixes and examples
    lines = answer_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip completely unwanted lines
        if (line in ["Human:", "Assistant:", "Legal Assistant:", "🤖 Legal Assistant:", "Answer:", "### Answer:"] or
            line.startswith("### Example") or 
            line.startswith("Question:") or
            line.startswith("Human: ") or 
            line.startswith("Assistant: ") or 
            line.startswith("Legal Assistant: ") or 
            line.startswith("🤖 Legal Assistant: ")):
            continue
            
        # Don't remove ** yet - we need to preserve them for bold formatting
        if line:  # Only add non-empty lines
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Define main section headers that should be styled with HTML bold
    main_section_headers = [
        'Direct Answer:',
        'Legal Basis:',
        'Key Differences:',
        'Additional Notes:',
        'Legal Implications:',
        'Practical Application:'
    ]
    
    # Convert to HTML with proper formatting
    html_lines = []
    for line in cleaned_text.split('\n'):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
            
        # Check if this line is a main section header (without ** markdown)
        clean_line = line.replace('**', '')
        is_main_header = any(clean_line.startswith(header) for header in main_section_headers)
        
        if is_main_header:
            # Format as bold header using HTML styling
            header_text = clean_line
            html_lines.append(f'<div style="font-weight: bold; color: #1f2937; font-size: 1.1em; margin-top: 20px; margin-bottom: 10px;">{header_text}</div>')
        else:
            # Regular content - convert **text** to <strong>text</strong>
            formatted_line = line
            
            # Convert markdown bold to HTML strong tags
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_line)
            
            # Check for bullet points
            if line.startswith('•') or line.startswith('*') or line.startswith('-'):
                html_lines.append(f'<div style="margin-bottom: 8px; margin-left: 20px; line-height: 1.6; color: #374151;">{formatted_line}</div>')
            else:
                # Regular text
                html_lines.append(f'<div style="margin-bottom: 8px; line-height: 1.6; color: #374151;">{formatted_line}</div>')
    
    return ''.join(html_lines)

def display_quick_actions():
    """Display quick action buttons for common legal topics"""
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏛️ Constitutional Rights", key="const_rights"):
            st.session_state.user_input = "What are the fundamental human rights protected by the Constitution of Zimbabwe?"
            st.rerun()
    
    with col2:
        if st.button("⚡ Criminal Law", key="criminal_law"):
            st.session_state.user_input = "What constitutes criminal conduct under Zimbabwean law?"
            st.rerun()
    
    with col3:
        if st.button("👶 Children's Rights", key="children_rights"):
            st.session_state.user_input = "At what age can a child be held criminally responsible in Zimbabwe?"
            st.rerun()
    
    with col4:
        if st.button("🔒 Arrest Procedures", key="arrest_proc"):
            st.session_state.user_input = "What rights does a person have if they are arrested?"
            st.rerun()

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>⚖️ ZimLaw</h2>
            <p style='color: #6b7280; margin: 0;'>AI Legal Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("🔄 New Conversation", key="new_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.user_input = ""
            st.rerun()
        
        st.markdown("---")
        
        # Example Questions
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What rights does a person have if they are arrested?",
            "Can a 6-year-old child be charged with a crime?",
            "What is the difference between intention and negligence?",
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💬 {question[:40]}{'...' if len(question) > 40 else ''}", 
                        key=f"example_{i}", 
                        use_container_width=True):
                st.session_state.user_input = question
                st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("""
        ### ℹ️ About
        ZimLaw Assistant is an AI-powered legal research tool trained on Zimbabwean law. 
        
        **Features:**
        - 📚 460+ legal documents
        - 🔍 Intelligent search
        - ⚖️ Accurate citations
        - 🎯 Professional format
        
        **Powered by:**
        - 🤖 Ollama LLM
        - 🔗 RAG Technology
        - 📖 Few-shot Learning
        """)
    
    # Main content
    st.markdown("""
    <div class='main-content'>
        <div class='header-container'>
            <h1 class='header-title'>⚖️ ZimLaw Assistant</h1>
            <p class='header-subtitle'>Your AI-powered legal research companion for Zimbabwean law</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    try:
        with st.spinner("🔧 Initializing legal assistant..."):
            rag_chain, conditioned_generator = initialize_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return
    
    # Quick actions
    display_quick_actions()
    
    # Chat interface
    st.markdown("### 💬 Ask a Legal Question")
    
    # User input
    user_question = st.text_area(
        "Ask your legal question",
        placeholder="Ask me anything about Zimbabwean law... 🤔\n\nExamples:\n• What are my rights if I'm arrested?\n• Can a 10-year-old be charged with a crime?\n• What constitutes criminal negligence?",
        height=120,
        key="question_input",
        label_visibility="hidden",
        value=st.session_state.user_input
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_clicked = st.button("🔍 Get Legal Answer", type="primary", use_container_width=True)
    
    # Process question
    if submit_clicked and user_question.strip():
        # Clear the input
        st.session_state.user_input = ""
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        try:
            # Show initial thinking message
            with st.spinner("🧠 Analyzing legal documents..."):
                # Get RAG sources for display (non-streaming part)
                rag_result = rag_chain.answer_question(user_question)
                sources = rag_result.get('sources', [])
            
            # Create a placeholder for the streaming assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "",
                "sources": sources,
                "metadata": {
                    "examples_used": 10,
                    "method": "few_shot_conditioning_stream"
                },
                "streaming": True,  # Flag to indicate this is being streamed
                "streaming_question": user_question  # Store the question for streaming
            })
            
            # Force a rerun to show the empty assistant message
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while processing your question: {str(e)}"
            })
    
    # Display conversation
    if st.session_state.messages:
        st.markdown("### 📖 Conversation")
        
        # Display messages in reverse order (newest first)
        for i, message in enumerate(reversed(st.session_state.messages)):
            # Add "Latest" indicator for the most recent message
            latest_indicator = "🔥 <small>Latest</small><br>" if i == 0 else ""
            
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    {latest_indicator}<strong>🙋 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # assistant
                # Check if this message is currently being streamed
                if message.get("streaming", False):
                    # This is the message we're currently streaming into
                    st.markdown(f"""
                    <div class='chat-message assistant-message'>
                        {latest_indicator}<strong>⚖️ Legal Assistant:</strong><br>
                        <div class='legal-answer'>
                    """, unsafe_allow_html=True)
                    
                    # Create streaming placeholder
                    message_placeholder = st.empty()
                    
                    # Get the question from the message
                    streaming_question = message.get("streaming_question", "")
                    
                    # Stream the response
                    full_response = ""
                    for chunk in conditioned_generator.generate_answer_stream(streaming_question):
                        full_response += chunk
                        # Update the message content in real-time
                        formatted_content = format_legal_answer(full_response)
                        message_placeholder.markdown(formatted_content, unsafe_allow_html=True)
                    
                    # Finalize the message
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Update the message in session state and remove streaming flag
                    message_index = len(st.session_state.messages) - 1 - i
                    st.session_state.messages[message_index]["content"] = full_response
                    st.session_state.messages[message_index]["streaming"] = False
                    del st.session_state.messages[message_index]["streaming_question"]
                    
                    # Rerun to show the final state
                    st.rerun()
                    
                else:
                    # Regular completed message
                    formatted_answer = format_legal_answer(message['content'])
                    
                    st.markdown(f"""
                    <div class='chat-message assistant-message'>
                        {latest_indicator}<strong>⚖️ Legal Assistant:</strong><br>
                        <div class='legal-answer'>
                            {formatted_answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show metadata
                if "metadata" in message:
                    metadata = message["metadata"]
                    st.markdown(f"""
                    <div class='metadata-container'>
                        📊 <strong>Analysis Details:</strong> Used {metadata.get('examples_used', 0)} legal examples | Method: {metadata.get('method', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show sources in expandable section
                if "sources" in message and message["sources"]:
                    # Use reversed index for unique identification
                    reversed_index = len(st.session_state.messages) - 1 - i
                    with st.expander(f"📚 Legal Sources ({len(message['sources'])} documents)", expanded=False):
                        for j, source in enumerate(message["sources"][:5]):  # Show top 5
                            st.markdown(f"""
                            <div class='source-card'>
                                <div class='source-title'>{source.get('act', 'Unknown Act')}</div>
                                <div class='source-details'>
                                    📖 Chapter: {source.get('chapter', 'N/A')} | 
                                    📋 Section: {source.get('section', 'N/A')}<br>
                                    📝 {source.get('title', 'No title available')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Legal disclaimer
    st.markdown("""
    <div class='warning-box'>
        <div class='warning-text'>
            ⚠️ <strong>Important Legal Disclaimer:</strong> This AI assistant provides legal information for educational purposes only. 
            It does not constitute legal advice. Always consult with a qualified legal professional for specific legal matters.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #6b7280; border-top: 1px solid #e5e7eb; margin-top: 3rem;'>
        <p>📚 Built with 460+ Zimbabwean legal documents | 🤖 Powered by AI | 🔍 Enhanced with RAG technology</p>
        <p><small>ZimLaw Assistant v2.0 | Made with ❤️ for legal research</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
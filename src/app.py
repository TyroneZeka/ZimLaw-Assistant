import streamlit as st
from utils.rag_chain import ZimLawRAGChain
import time
from typing import Dict, Any
from PIL import Image

@st.cache_resource
def initialize_rag_chain() -> ZimLawRAGChain:
    """Initialize and cache the RAG chain"""
    return ZimLawRAGChain()

# Custom CSS for dark theme UI
st.markdown("""
<style>
    /* Dark theme styling */
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    
    .stTextInput, .stTextArea {
        background-color: #333;
        color: white;
        border-radius: 8px;
    }
    
    .source-item {
        background-color: #333;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .chat-container {
        background-color: #333;
        border-radius: 8px;
        padding: 16px;
        margin: 24px auto;
        color: white;
    }
    
    .action-button {
        background-color: #333;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        color: white;
        font-size: 14px;
        margin: 4px;
    }
    
    .sidebar {
        background-color: #1E1E1E;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Page config
    st.set_page_config(
        page_title="ZimLaw Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        # Logo and title
        try:
            logo_image = Image.open("./screenshots/logo.png")
            st.image(logo_image, width=50)
        except:
            st.markdown("⚖️")  # Fallback if logo not found
            
        st.markdown("<h3 style='text-align: center; color: white;'>ZimLaw Assistant</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Powered by Deepseek</p>", unsafe_allow_html=True)

        # New Chat Button
        if st.button("New Chat", key="new_chat_button"):
            st.session_state.clear()

        # Search Bar
        st.text_input("Search Previous Chats", placeholder="Search", key="search_input")

        # Chat History
        st.markdown("<h4 style='color: white;'>Recent Conversations</h4>", unsafe_allow_html=True)
        st.markdown("- Rights of Arrested Persons")
        st.markdown("- Employment Contract Terms")
        st.markdown("- Property Rights in Zimbabwe")

        # User Profile
        st.markdown("---")
        try:
            user_profile_image = Image.open("user_profile.png")
            st.image(user_profile_image, width=30)
        except:
            st.markdown("👤")  # Fallback if profile image not found
        st.markdown("<p style='color: white;'>Legal Assistant</p>", unsafe_allow_html=True)

    # Main Content Area
    try:
        rag_chain = initialize_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize the legal assistant: {str(e)}")
        return

    # Greeting and Main Interface
    st.markdown("<h2 style='text-align: center; color: white;'>Your AI Legal Assistant</h2>", unsafe_allow_html=True)

    # Message Input
    user_question = st.text_area(
        "",  # Remove label
        placeholder="Ask me anything about Zimbabwean law...",
        height=150,
        width=2000,
        key="user_input"
    )

    # Interactive Elements
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.markdown("<div class='action-button'>Criminal Law</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='action-button'>Civil Rights</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='action-button'>Labour Law</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='action-button'>More</div>", unsafe_allow_html=True)

    # Process Question
    if user_question:
        with st.container():
            st.markdown(f"""
            <div class='chat-container'>
                <p><strong>You:</strong> {user_question}</p>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner(""):
                    result = rag_chain.answer_question(user_question)
                    
                    # Display answer in chat style
                    st.markdown(f"""
                    <div class='chat-container'>
                        <p><strong>Assistant:</strong> {result["answer"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sources in collapsible
                    with st.expander("📚 View Sources", expanded=False):
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
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        ⚠️ This is an AI-powered legal information tool. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
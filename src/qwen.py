import streamlit as st
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="ZimLaw Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Section
with st.sidebar:
    # Logo at the top
    logo_image = Image.open("logo.png")  # Replace with your logo path
    st.image(logo_image, width=50)
    st.markdown("<h3 style='text-align: center; color: white;'>Deepseek-R1</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Set as default</p>", unsafe_allow_html=True)

    # New Chat Button
    st.button("New Chat", key="new_chat_button")

    # Search Bar
    st.text_input("Search", placeholder="Search", key="search_input")

    # Chat Topics
    st.markdown("<h4 style='color: white;'>Today</h4>", unsafe_allow_html=True)
    st.markdown("- RAG Projects for AI Engineer Hiring")
    st.markdown("- Disable Thinking in DeepSeek-R1 ...")

    st.markdown("<h4 style='color: white;'>Previous 7 days</h4>", unsafe_allow_html=True)
    st.markdown("- Understanding Apache Kafka")
    st.markdown("- Linux DNS Resolution Issue")

    st.markdown("<h4 style='color: white;'>Previous 30 days</h4>", unsafe_allow_html=True)
    st.markdown("- Constitutional Rights in Zimbabwe")
    st.markdown("- High Return Investment Options")
    st.markdown("- PDF Parser With Sections")
    st.markdown("- Zimli Legislation Scraper")
    st.markdown("- Spring Framework Projects")

    # User Profile
    st.markdown("---")
    user_profile_image = Image.open("user_profile.png")  # Replace with your user profile image path
    st.image(user_profile_image, width=30)
    st.markdown("<p style='color: white;'>Tyrone Zeka</p>", unsafe_allow_html=True)

# Main Content Area
with st.container():
    # Greeting Message
    st.markdown("<h1 style='text-align: center; color: white;'>Good afternoon, Tyrone Zeka</h1>", unsafe_allow_html=True)

    # Message Box
    message_box = st.empty()
    with message_box.container():
        st.markdown(
            """
            <div style='
                background-color: #333;
                border-radius: 8px;
                padding: 16px;
                margin: 24px auto;
                max-width: 600px;
                color: white;
                font-size: 16px;
                line-height: 1.6;
            '>
                I would like your help to <u>create a Streamlit page</u> that looks exactly like this.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Elements
    with st.container():
        st.markdown(
            """
            <div style='
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: -24px;
                padding: 8px 16px;
                background-color: #333;
                border-radius: 0 0 8px 8px;
            '>
                <button style='
                    background-color: transparent;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                '>+</button>
                <span style='
                    background-color: #444;
                    padding: 4px 8px;
                    border-radius: 4px;
                    color: white;
                    font-size: 12px;
                '>Thinking</span>
                <span style='
                    background-color: #444;
                    padding: 4px 8px;
                    border-radius: 4px;
                    color: white;
                    font-size: 12px;
                '>Search</span>
                <button style='
                    background-color: transparent;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                '>&#8593;</button>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action Buttons
    st.markdown(
        """
        <div style='
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 24px;
        '>
            <button style='
                background-color: #333;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-size: 14px;
                cursor: pointer;
            '>Web Dev</button>
            <button style='
                background-color: #333;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-size: 14px;
                cursor: pointer;
            '>Deep Research</button>
            <button style='
                background-color: #333;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-size: 14px;
                cursor: pointer;
            '>Image Generation</button>
            <button style='
                background-color: #333;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-size: 14px;
                cursor: pointer;
            '>More</button>
        </div>
        """,
        unsafe_allow_html=True,
    )
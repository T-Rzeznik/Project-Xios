import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gpt
from functions import *
from authenticator import authenticator

load_dotenv('keys.env')

# Configure Streamlit page settings for better user interface
st.set_page_config(
    page_title="Linux Lab Generator",
    page_icon=":robot_face:",  # Favicon emoji for browser tab
    layout="wide",  # Uses full screen width for better content display
)

# Retrieve API key from environment variables for security
#API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google's Gemini-Pro AI model with API key
gpt.configure(api_key=API_KEY)

model = gpt.GenerativeModel('gemini-1.5-pro-latest')
light_model = gpt.GenerativeModel('gemini-1.5-flash-001')


def main():
    st.markdown(
        """
        <style>
        .header {
            background-color: #CC0025;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        <div class="header">
            Commands Generator
        </div>
        """,
        unsafe_allow_html=True
    )

    authenticator(model)
    
    # Check if the user is logged in
    if 'username' not in st.session_state:
        st.warning("Please log in to access the Linux Lab Generator.")
        return  # Exit the main function if not logged in

    # Initialize or retrieve existing chat session from Streamlit's session state
    initialize_chat_session(model)

    # Create tabs for selecting between Lab Generator and Upload Labs
    tab1, tab2 = st.tabs(["Lab Generator", "Upload Labs"])

    with tab1:
        # Lab Generator functionality
        st.sidebar.subheader("Select Chat Generation Mode")
        chat_mode = st.sidebar.selectbox(
            "Choose a mode:",
            ["Prompt Engineering", "Zero-shot Context Comparison", "Vectorized Database", "Voice"],
            index=0  # Set default to "Prompt Engineering"
        )

        # Store the selected mode in session state
        st.session_state.chat_mode = chat_mode


        # Main application title display
        st.title("🤖 Linux Lab Generator")
        with st.sidebar:
            st.title("Lab Sessions")
            manage_sessions(model)

        # Create tabs for different functionalities
        initialize_display_history()
        create_lab_generation_form(light_model)
        display_generated_lab_info(light_model)
        display_chat_history()
        handle_user_input()

    with tab2:
        # Upload Labs functionality
        st.title("Upload Labs")
        st.write("Here you can upload your existing lab files.")
        
        # Call the upload and process function
        upload_and_process_file()
        if st.button("Delete All Data from Vector Database"):
            delete_all_data_from_vector_db()

if __name__ == "__main__":
    main()
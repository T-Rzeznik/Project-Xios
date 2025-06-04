import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
import pandas as pd
from topic_classifier import is_on_topic
import pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
import markdown
from bs4 import BeautifulSoup
import re
import pdfplumber

load_dotenv('keys.env')

#TTS work
import google.generativeai as gpt
import pygame
from openai import OpenAI
import speech_recognition as sr
import pyaudio


#PDF gen
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO


mongo_uri = os.getenv("MONGODB_URI")
pinecone_api_key =  os.getenv("PINECONE_KEY")

def get_database():
    client = MongoClient(mongo_uri)
    return client['ragbot']

def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role
    
def generate_keywords(model):
    generated_lab = st.session_state.get('generated_lab', None)
    if generated_lab:
        prompt = (f"Generate a list 5-10 keywords that describe the following Lab content: {generated_lab}"
                 " DO NOT USE SYMBOLS IN KEYWORDS UNLESS NECESSARY, ONLY RESPOND WITH KEYWORDS SEPARATED BY COMMAS,"
                 " DO NOT INCLUDE EXTRA WORDS OR SYMBOLS")
        try:
            response = model.generate_content(prompt)
            
            # Split the response by commas and clean each keyword
            raw_keywords = response.text.replace('\n', ',').split(',')
            keywords = [k.strip().lower() for k in raw_keywords if k.strip()]
            
            print(f"Generated keywords: {keywords}") #Debugging line.
            return keywords
        except Exception as e:
            print(f"Error generating keywords: {e}")
            return ["Error"] #Ensures a list is returned.
    else:
        return ["none2"] #Ensures a list is returned.

def fetch_gemini_response(user_query):
    try:
        response = st.session_state.chat_session.send_message(user_query)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error generating a response. Please try again."

def save_session(session_name, chat_history):
    """
    Saves the chat session to MongoDB with proper formatting.
    The chat_history parameter can be either:
    1. A list of Content objects (from Gemini's chat session)
    2. A list of dictionaries (from our display history)
    """
    db = get_database()
    sessions_collection = db['sessions']
    
    # Get the current user's username
    username = st.session_state.get('username', 'guest')  # Default to 'guest' if not logged in
    
    # Format the history based on its type
    formatted_history = []
    
    if chat_history and isinstance(chat_history[0], dict):
        formatted_history = chat_history
    else:
        for message in chat_history:
            role = message.role
            content = message.parts[0].text
            
            formatted_history.append({
                "role": role,
                "content": content
            })
    
    # Include the generated lab in the session data
    generated_lab = st.session_state.get('generated_lab', None)
    generated_keywords = st.session_state.get('generated_keywords', [])
    lab_summary = st.session_state.get('lab_summary', None)
    
    try:
        sessions_collection.update_one(
            {"session_name": session_name, "username": username},  # Include username in the query
            {
                "$set": {
                    "session_name": session_name,
                    "chat_history": formatted_history,
                    "last_modified": datetime.datetime.utcnow(),
                    "username": username,  # Save the username with the session
                    "generated_lab": generated_lab,  # Save the generated lab
                    "generated_keywords": generated_keywords,
                    "lab_summary": lab_summary
                }
            },
            upsert=True
        )
        print("***Uploaded to MongoDB Success***")
        return True
    except Exception as e:
        st.error(f"An error occurred while saving the session: {e}")
        return False

def load_session(session_name, model):
    db = get_database()
    sessions_collection = db['sessions']
    
    # Get the current user's username
    username = st.session_state.get('username', 'guest')  # Default to 'guest' if not logged in
    
    session = sessions_collection.find_one({"session_name": session_name, "username": username})  # Include username in the query
    
    if not session:
        return model.start_chat(history=[]), [], None  # Return None for generated_lab if session not found
    
    gemini_messages = []
    for message in session['chat_history']:
        if message['role'] == 'user':
            gemini_messages.append({
                'role': 'user',
                'parts': [{'text': message['content']}]
            })
        else:
            gemini_messages.append({
                'role': 'model',
                'parts': [{'text': message['content']}]
            })
    
    chat = model.start_chat(history=gemini_messages)
    generated_lab = session.get('generated_lab', None)  # Load the generated lab from the session
    generated_keywords = session.get('generated_keywords', [])  # Load the generated keywords from the session
    lab_summary = session.get('lab_summary', None) 

    st.session_state.generated_keywords = generated_keywords
    st.session_state.lab_summary = lab_summary

    return chat, session['chat_history'], generated_lab  # Return the generated lab


def get_saved_sessions():
    db = get_database()
    sessions_collection = db['sessions']
    
    # Get the current user's username
    username = st.session_state.get('username', 'guest')  # Default to 'guest' if not logged in
    
    sessions = sessions_collection.find({"username": username}, {"session_name": 1, "last_modified": 1}).sort("last_modified", -1)
    return [session['session_name'] for session in sessions]

def get_public_labs():
    db = get_database()
    public_labs_collection = db['public_labs']
    public_labs = public_labs_collection.find().sort("shared_at", -1)  # Sort by 'shared_at' in descending order
    return [lab['lab_query'] for lab in public_labs]

def delete_session(session_name):
    db = get_database()
    sessions_collection = db['sessions']
    result = sessions_collection.delete_one({"session_name": session_name})
    return result.deleted_count > 0

def initialize_chat_session(model):
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.is_public_lab = False  # Initialize is_public_lab to False

def reset(model):
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.display_history = []
    st.session_state.current_session = None
    st.session_state.lab_generated = False
    st.session_state.generated_lab = None
    st.session_state.is_public = False
    st.session_state.generated_keywords = None
    st.session_state.lab_summary = None
    if 'lab_query' in st.session_state:
        del st.session_state.lab_query
    st.rerun()

def manage_sessions(model):
    if st.button("Start New Session"):
        reset(model)

    view_tab, delete_tab, public_tab = st.tabs(["View History", "Delete History", "Public Labs"])
    saved_sessions = get_saved_sessions()
    public_labs = get_public_labs()

    with view_tab:
        st.subheader("History")
        for session in saved_sessions:
            if st.button(session, key=f"view_{session}"):
                new_chat, history, generated_lab = load_session(session, model)
                st.session_state.chat_session = new_chat
                st.session_state.display_history = history
                st.session_state.current_session = session
                st.session_state.lab_generated = True
                st.session_state.is_public_lab = False
                st.session_state.generated_lab = generated_lab
                st.rerun()

    with delete_tab:
        st.subheader("Select Sessions to Delete")
        for session in saved_sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(session)
            with col2:
                if st.button("❌", key=f"delete_{session}_{st.session_state['username']}"):
                    delete_session(session)
                    st.rerun()

    with public_tab:
        st.subheader("Public Labs")
        for index, lab_query in enumerate(public_labs):
            if isinstance(lab_query, str) and lab_query:
                if st.button(lab_query, key=f"view_{lab_query}_{index}"):
                    load_public_session(lab_query)  # Load the public lab session
                    st.session_state.lab_generated = True
                    st.session_state.is_public_lab = True
                    st.rerun()
            else:
                st.error("Invalid lab query detected.")
        

def initialize_display_history():
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
    if "selected_distro" not in st.session_state:  # Initialize selected_distro
        st.session_state.selected_distro = None  # or set a default value if needed

def create_lab_generation_form(model):
    if not st.session_state.get('lab_generated', False):  # Check if lab has not been generated
        with st.form(key='lab_generation_form'):
            st.subheader("Lab Configuration")
            linux_distros = ["Debian", "Arch", "Kali", "Ubuntu", "Fedora", "CentOS"]
            selected_distro = st.selectbox("Select a Linux Distribution", linux_distros)
            lab_query = st.text_input("Generate Lab Query")
            submit_button = st.form_submit_button("Generate Lab")

            if submit_button and lab_query:
                prompt = f"Please create a linux lab manual for {selected_distro} with respect to {lab_query} using detailed commands, options, and explanations step by step."
                lab_response = fetch_gemini_response(prompt)
                
                # Store in display history without including the generated lab
                st.session_state.generated_lab = lab_response
                st.session_state.lab_generated = True
                st.session_state.selected_distro = selected_distro  # Store the selected distro
                st.session_state.display_history = []  # Clear display history or keep it empty

                keywords = generate_keywords(model)
               
                st.session_state.generated_keywords = keywords
                print("KEYWORDS SESSION STATE: ")
                print( "------------------ ---------------------")
                print(st.session_state.generated_keywords)

                # Save the session using display history
                save_session(lab_query, st.session_state.display_history)
                st.session_state.current_session = lab_query
                username = st.session_state.get('username', 'guest')  # Default to 'guest' if not logged in

                html = markdown.markdown(lab_response)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                
                text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
                text = re.sub(r'`.*?`', '', text)  # Remove inline code
                text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Remove links
                text = re.sub(r'#{1,6}\s.*?\n', '', text)  # Remove headers
                text = text.strip()

                # Upsert embeddings into Pinecone after lab generation
                texts = [text]
            
                embeddings = embedding_model.encode(texts)

                for i, text in enumerate(texts):
                    metadata = {"text": text, "creator": username, "keywords": keywords}
                    
                    index.upsert([(lab_query, embeddings[i], metadata)], namespace=username)  # Use lab_query as the ID
                    print(f"Successful Upsert, Data: ID: {lab_query}, Vector: {embeddings[i]}, Metadata: {metadata}")
                
                st.rerun()

def display_chat_history():
    for msg in st.session_state.display_history:
        with st.chat_message(map_role(msg["role"])):
            st.markdown(msg["content"])

def wave_to_text():
    r = sr.Recognizer()
    # Configure the recognizer for longer speech
    r.phrase_threshold = 0.3  # More sensitive to continuous speech
    r.pause_threshold = 2.0   # Allow 2 second pauses before stopping
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        # Listen without timeout for longer phrases
        audio = r.listen(source, phrase_time_limit=None)

    try:
        print("Recognizing...")
        text = r.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def speak(text):
    openai_api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=openai_api_key)

    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"
        )

        # Create a BytesIO object from the audio content
        audio_buffer = BytesIO(response.content)
        
        # Load and play the audio
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"Error during TTS: {e}")
    
def listen_for_wake_word():
    wake_word = "gemini"
    print("Listening for wake word...")
    while True:
        try:
            user_input = wave_to_text()
            if user_input == "exit":
                print("Exiting program...")
                return False
            if wake_word in user_input:
                print("Wake word detected!")
                return prompt_gpt()
        except Exception as e:
            print(f"Wake word listener error: {e}")
            continue

def prompt_gpt():
    generation_config = gpt.GenerationConfig(
    temperature=0.9,  # Increased from 0.7 for more creativity
    max_output_tokens=150,  # Increased token limit for more expressive responses
    top_p=0.95,  # Increased for more diverse word choices
    top_k=60,  # Increased for broader vocabulary selection
    candidate_count=1
)
    while True:
        try:
            print("How can I help you? (say 'stop' to go back to wake word mode, 'exit' to quit)")
            user_input = wave_to_text()
            
            if not user_input:
                continue
                
            if user_input == "stop":
                print("Going back to wake word mode...")
                return True
            if user_input == "exit":
                print("Exiting program...")
                return False
            
            prompt = (
                            "You are a Linux Lab Generator AI designed to help users learn and troubleshoot with the generated Linux labs. Follow these guidelines:\n\n"
                            "2. **Guide Users:**\n"
                            "  - Assist users with their Linux labs by explaining related concepts or troubleshooting steps.\n"
                            "  - Provide concise, clear instructions or code snippets relevant to their lab or Linux-related questions.\n\n"
                            "3. **Enforce Topic Relevance:**\n"
                            "  - Respond only to questions related to Linux labs, Linux administration, the current lab, the chat history or related concepts.\n"
                            "  - If a question is off-topic, respond with the following error message:\n"
                            "    - \"Error: This tool is specifically designed to assist with Linux labs and related Linux concepts. Please ask a question relevant to these topics.\"\n\n"
                            "4. **Clarify User Requests:**\n"
                            "  - If a user request is unclear or lacks detail, ask for clarification. For example:\n"
                            "    - \"Can you provide more details about the type of Linux lab you'd like to create? For example, networking, scripting, or user management?\"\n\n"
                            "5. **Maintain Professional Tone:**\n"
                            "  - Always use a professional and helpful tone to guide users.\n\n"
                            "Your primary goal is to generate Linux labs and assist with Linux-related topics efficiently while keeping the conversation focused and relevant."
                            "MAKE SURE YOUR RESPONSE IS SHORT, Maximum 3 sentences"
                        )
            generated_lab = st.session_state.get('generated_lab', '')
            chat_history = "\n".join([f"User: {msg['content']}" for msg in st.session_state.display_history if msg['role'] == 'user'])
            full_prompt = f"{prompt}\n\nUser Input: {user_input}\n\nCurrent Lab: {generated_lab}\n\nChat History:\n{chat_history}"
            response = st.session_state.chat_session.send_message(full_prompt)
                
            gemini_response = response.text
            print("Gemini:", gemini_response)

            # Add user message and response to session state display history
            st.session_state.display_history.append({"role": "user", "content": user_input})
            st.session_state.display_history.append({"role": "model", "content": gemini_response})
            
            # Save session if there's a current session
            if hasattr(st.session_state, 'current_session') and st.session_state.current_session:
                save_session(st.session_state.current_session, st.session_state.display_history)


            speak(gemini_response)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Going back to wake word mode...")
            return True

def start_listening():
    print("Gemini Voice Assistant started. Say 'Gemini' to activate.")
    print("You can say 'exit' at any time to quit the program.")
    
    if st.session_state.voice_active == False:
        return

    continue_listening = True
    while continue_listening:
        continue_listening = listen_for_wake_word()

    st.rerun()


def handle_user_input():
    # Only show the chat input if not in public lab mode
    if st.session_state.get('lab_generated', False) and not st.session_state.get('is_public_lab', False):
        chat_mode = st.session_state.get('chat_mode', 'Prompt Engineering')

        if chat_mode == "Voice":
            # Voice Mode Activation
            st.markdown("### 🎤 Voice Mode")
            st.markdown("Click the button below to activate voice interaction")

            if st.button("🎤 Activate Voice Assistant", type="primary"):
                st.session_state.voice_active = True # Add state to track if voice is active

                if st.button("🛑 Stop Voice"):
                    st.session_state.voice_active = False # Set voice_active to false.
                    st.info("Voice interaction stopped.")
                   

                with st.spinner("Voice mode activated. Say 'Gemini' to wake up the assistant..."):
                    try:
                        voice_activated = start_listening()
                        if voice_activated:
                            st.success("Voice interaction completed")
                        else:
                            st.info("Voice mode exited")
                    except Exception as e:
                        st.error(f"Error in voice mode: {e}")
                        st.info("Please make sure your microphone is connected and permissions are granted")

                st.session_state.voice_active = False # Reset state after voice is done

                    

            with st.expander("Voice Assistant Instructions", expanded=False):
                st.markdown("""
                    ### How to use Voice Mode:
                    1. Click the 'Activate Voice Assistant' button above
                    2. Say "Gemini" to wake up the assistant
                    3. After activation, speak your question clearly
                    4. Say "stop" to return to wake word mode
                    5. Say "exit" to quit voice interaction
                    6. Click "Stop Voice" to manually end the voice session.

                    Note: Make sure your microphone is enabled and permissions are granted.
                    """)
        else:
            # Regular Chat Input Mode
            user_input = st.chat_input("Ask about the lab...")
            if user_input:
                if st.session_state.get('current_session'):
                    generated_lab = st.session_state.get('generated_lab', '')
                    if not generated_lab:
                        st.error("Generated lab is not available.")
                        return

                st.chat_message("user").markdown(user_input)

                try:
                    # response based on chat_mode (excluding Voice)
                    if chat_mode == "Prompt Engineering":
                        prompt = (
                            "You are a Linux Lab Generator AI designed to help users learn and troubleshoot with the generated Linux labs. Follow these guidelines:\n\n"
                            "2. **Guide Users:**\n"
                            "  - Assist users with their Linux labs by explaining related concepts or troubleshooting steps.\n"
                            "  - Provide concise, clear instructions or code snippets relevant to their lab or Linux-related questions.\n\n"
                            "3. **Enforce Topic Relevance:**\n"
                            "  - Respond only to questions related to Linux labs, Linux administration, the current lab, the chat history or related concepts.\n"
                            "  - If a question is off-topic, respond with the following error message:\n"
                            "    - \"Error: This tool is specifically designed to assist with Linux labs and related Linux concepts. Please ask a question relevant to these topics.\"\n\n"
                            "4. **Clarify User Requests:**\n"
                            "  - If a user request is unclear or lacks detail, ask for clarification. For example:\n"
                            "    - \"Can you provide more details about the type of Linux lab you'd like to create? For example, networking, scripting, or user management?\"\n\n"
                            "5. **Maintain Professional Tone:**\n"
                            "  - Always use a professional and helpful tone to guide users.\n\n"
                            "Your primary goal is to generate Linux labs and assist with Linux-related topics efficiently while keeping the conversation focused and relevant."
                        )
                        generated_lab = st.session_state.get('generated_lab', '')
                        chat_history = "\n".join([f"User: {msg['content']}" for msg in st.session_state.display_history if msg['role'] == 'user'])
                        full_prompt = f"{prompt}\n\nUser Input: {user_input}\n\nCurrent Lab: {generated_lab}\n\nChat History:\n{chat_history}"
                        response = st.session_state.chat_session.send_message(full_prompt)

                    elif chat_mode == "Zero-shot Context Comparison":
                        if is_on_topic(user_input, st.session_state.current_session, generated_lab):
                            response = st.session_state.chat_session.send_message(user_input)
                            if response is None:
                                st.error("No response received from the chat session.")
                        else:
                            st.warning("Your message appears to be off-topic for Zero-shot Context Comparison.")
                            return
                    
                    elif chat_mode == "Vectorized Database":
                        is_relevant, context_texts = query_vectorized_db(user_input)
                        if is_relevant:
                            augmented_prompt = f"""Consider the following relevant context from previous labs:{' '.join(context_texts)} 
                            Now please answer this user question:{user_input} Please use any relevant information from the context to enhance your response."""
                        response = st.session_state.chat_session.send_message(augmented_prompt)

                    elif chat_mode == "Vectorized Database":
                        print("\n=== Starting Vectorized Database Mode ===")
                        is_relevant, context_texts = query_vectorized_db(user_input)
                        
                        if is_relevant:
                            print("\nCreating augmented prompt with context...")
                            augmented_prompt = f"""Consider the following relevant context from previous labs:{' '.join(context_texts)} 
                            Now please answer this user question:{user_input} Please use any relevant information from the context to enhance your response."""
                            
                            print(f"Augmented prompt length: {len(augmented_prompt)} characters")
                            print("Sending to Gemini...")
                            
                            response = st.session_state.chat_session.send_message(augmented_prompt)
                            print("Response received from Gemini")
                        else:
                            print("No relevant matches found - query deemed off-topic")
                            st.warning("Your message appears to be off-topic for Vectorized Database.")
                            return

                    response_text = response.text

                    st.chat_message("assistant").markdown(response_text)
                    st.session_state.display_history.append({"role": "user", "content": user_input})
                    st.session_state.display_history.append({"role": "model", "content": response_text})

                    

                    if hasattr(st.session_state, 'current_session') and st.session_state.current_session:
                        save_session(st.session_state.current_session, st.session_state.display_history)
                    
                    st.rerun()
                    
                    

                except Exception as e:
                    st.error(f"Error processing your message: {e}")

def share_lab(lab_query, selected_distro, lab_response):
    db = get_database()
    public_labs_collection = db['public_labs']
    
    # Get the current user's username
    username = st.session_state.get('username', 'guest')  # Default to 'guest' if not logged in
    
    try:
        public_labs_collection.insert_one({
            "lab_query": lab_query,
            "selected_distro": selected_distro,  # Save the selected distribution
            "lab_response": lab_response,
            "shared_at": datetime.datetime.utcnow(),
            "shared_by": username  # Store the username of the sharer
        })
        
        st.success("Lab shared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error sharing the lab: {e}")


def generate_lab_summary(model):
    generated_lab = st.session_state.get('generated_lab', None)

    try:
        prompt = (
            "Please provide a brief summary of this Linux lab that explains: \n"
            "1. What the lab is about\n"
            "2. What skills or concepts the user will learn\n"
            f"\nLab content: {generated_lab}"
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"
    



def display_generated_lab_info(model):
    if st.session_state.get('lab_generated', False):
        pdf_buffer = generate_pdf(st.session_state.generated_lab)
        with st.expander("Generated Lab Information", expanded=True):
            st.markdown("**Lab Query:**")
            st.markdown(f"{st.session_state.current_session}")
            st.markdown("**Selected Linux Distribution:**")
            st.markdown(f"{st.session_state.selected_distro}")
            generated_keywords = st.session_state.get('generated_keywords', [])

            if generated_keywords:
                st.markdown("**Keywords generated from lab:**")
                st.markdown(", ".join(generated_keywords))
            # Display who shared the lab if it's a public lab
            if st.session_state.get('is_public_lab', True):
                st.markdown("**Shared By:**")
                st.markdown(f"{st.session_state.get('shared_by', 'Unknown')}")
            
                
        if not st.session_state.get('is_public_lab', False):
                
                

                if st.button("Share Lab"):
                    share_lab(st.session_state.current_session, st.session_state.selected_distro, st.session_state.generated_lab)

                st.download_button(
                    label="Click here to download PDF",
                    data=pdf_buffer,
                    file_name=f"{st.session_state.current_session}.pdf",
                    mime="application/pdf"
                )    

        if st.button("📝 Generate Quick Summary"):
                summary = generate_lab_summary(model)
                st.session_state.lab_summary = summary
                save_session(st.session_state.current_session, st.session_state.display_history)

        if st.session_state.get('lab_summary'):
            with st.expander("Summarized Lab", expanded=False):
                    st.markdown("**Lab Summary:**")
                    st.markdown(st.session_state.lab_summary)

            
                    
        
        st.markdown(f"{st.session_state.generated_lab}")
            
        
                

def load_public_session(session_name):
    db = get_database()
    public_labs_collection = db['public_labs']
    
    # Fetch the public lab session from the database
    public_lab = public_labs_collection.find_one({"lab_query": session_name})
    
    if not public_lab:
        st.error("Public lab session not found.")
        return
    
    # Clear the history session
    st.session_state['display_history'] = []
    st.session_state['current_session'] = public_lab['lab_query']
    st.session_state['selected_distro'] = public_lab.get('selected_distro', 'Unknown')
    st.session_state['generated_lab'] = public_lab['lab_response']
    st.session_state['is_public_lab'] = True
    st.session_state['shared_by'] = public_lab.get('shared_by', 'Unknown')
    
    
    
        


pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

# index name and embedding dimension
index_name = "cslab"  
dimension = 384  


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Get the host of the created index
host = pc.describe_index(index_name).host


index = pc.Index(index_name, host=host)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_vectorized_db(user_input):
    """
    Query the vectorized database and return both relevance and matched content
    """
    print("\n=== Starting Vectorized DB Query ===")
    print(f"User Input: {user_input}")
    
    username = st.session_state.get('username', 'guest')
    print(f"Current User: {username}")
    
    # Generate embedding for user input
    user_embedding = embedding_model.encode(user_input)
    print(f"Generated embedding shape: {user_embedding.shape}")
    
    # Query Pinecone
    print("\nQuerying Pinecone database...")
    results = index.query(
        vector=user_embedding.tolist(), 
        top_k=3, 
        include_metadata=True, 
        namespace=username
    )
    
    print(f"\nReceived {len(results.get('matches', []))} matches from Pinecone")
    
    threshold = 0.3
    print(f"Similarity threshold: {threshold}")
    
    # Process matches
    relevant_matches = [
        match for match in results.get("matches", [])
        if match.get("score", 0) >= threshold
    ]
    
    print(f"\nFound {len(relevant_matches)} relevant matches above threshold")
    
    # Extract relevant texts
    context_texts = []
    for i, match in enumerate(relevant_matches, 1):
        print(f"\nMatch #{i}:")
        print(f"- Score: {match.get('score', 0):.4f}")
        print(f"- ID: {match.get('id', 'N/A')}")
        if match.get("metadata") and match["metadata"].get("text"):
            context_texts.append(match["metadata"]["text"])
            print(f"- Text length: {len(match['metadata']['text'])} characters")
            print(f"- Preview: {match['metadata']['text'][:100]}...")
    
    print(f"\nTotal context texts collected: {len(context_texts)}")
    print("=== Vectorized DB Query Complete ===\n")
    
    return bool(relevant_matches), context_texts

def query_vectorized_db(user_input):
    """
    Query the vectorized database and return both relevance and matched content
    """
    username = st.session_state.get('username', 'guest')
    user_embedding = embedding_model.encode(user_input)
    results = index.query(
        vector=user_embedding.tolist(), 
        top_k=3, 
        include_metadata=True, 
        namespace=username
    )
    threshold = 0.3
    relevant_matches = [
        match for match in results.get("matches", [])
        if match.get("score", 0) >= threshold
    ]
    context_texts = []
    for match in relevant_matches:
        if match.get("metadata") and match["metadata"].get("text"):
            context_texts.append(match["metadata"]["text"])
    return bool(relevant_matches), context_texts



def generate_pdf(markdown_text): #Turn Generated_lab into PDF, with reportlab library

    buffer = BytesIO() #memory buffer to store pdf file
    doc = SimpleDocTemplate(buffer, pagesize=letter) #create the pdf
    styles = getSampleStyleSheet() #load styling for pdf

    #Custom styling
    styles.add(ParagraphStyle(name="Header1", fontSize=18, spaceAfter=10, textColor=colors.black, bold=True))
    styles.add(ParagraphStyle(name="Header2", fontSize=16, spaceAfter=8, textColor=colors.darkgray, bold=True))
    styles.add(ParagraphStyle(name="Bold", fontSize=12, spaceAfter=4, textColor=colors.black, bold=True))

    content = []

    for line in markdown_text.split("\n"):  # Process each line
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        line = re.sub(r"<i>(.*?)</i>", r"\1", line)  # Remove <i> tags inside words
        line = line.replace("<", "&lt;").replace(">", "&gt;")  # Escape angle brackets

        # Headers
        if line.startswith("# "):  # Markdown H1
            content.append(Paragraph(line[2:], styles["Header1"]))
        elif line.startswith("## "):  # Markdown H2
            content.append(Paragraph(line[3:], styles["Header2"]))

        # Bold and Italics
        elif "**" in line:  # Markdown bold
            line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            content.append(Paragraph(line, styles["Bold"]))
        elif "_" in line:  # Markdown italics
            line = re.sub(r"_(.*?)_", r"<i>\1</i>", line)
            content.append(Paragraph(line, styles["Normal"]))

        # Inline Code
        elif "`" in line:
            line = re.sub(r"`(.*?)`", r"<font face='Courier'><b>\1</b></font>", line)
            content.append(Paragraph(line, styles["Normal"]))

        # Unordered Lists
        elif line.startswith("- "):
            bullet_list = ListFlowable(
                [ListItem(Paragraph(line[2:], styles["Normal"]))],
                bulletType="bullet",
            )
            content.append(bullet_list)

        # Numbered Lists
        elif re.match(r"\d+\.", line):  # Matches "1. Item"
            numbered_list = ListFlowable(
                [ListItem(Paragraph(line[3:], styles["Normal"]))],
                bulletType="1",
            )
            content.append(numbered_list)

        else:
            content.append(Paragraph(line, styles["Normal"]))

        content.append(Spacer(1, 8))  # Add spacing
        
    doc.build(content)
    buffer.seek(0)
    return buffer

def upload_and_process_file():
    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        # Read the content based on file type
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Display cleaned text to the user
        st.write("Cleaned Text:")
        st.write(cleaned_text)
        
        # Upsert embeddings into Pinecone using SentenceTransformer
        embeddings = embedding_model.encode([cleaned_text])
        
        # Debugging: Print the generated vector
        print(f"Generated vector for '{uploaded_file.name}': {embeddings[0]}")
        print(cleaned_text)
        
        username = st.session_state.get('username', 'guest')
        # Include cleaned_text in the metadata
        index.upsert([(uploaded_file.name, embeddings[0].tolist(), {"creator": username, "text": cleaned_text})], namespace=username)
        
        # Debugging: Print the uploaded vector
        print(f"Uploaded vector for '{uploaded_file.name}' to Pinecone.")

def clean_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def delete_all_data_from_vector_db():
    """
    Deletes all data from the vector database.
    """
    try:
        index.delete(delete_all=True, namespace=st.session_state.get('username', 'guest'))  # Use the current user's namespace
        st.success("All data in the vector database has been deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting data from the vector database: {e}")
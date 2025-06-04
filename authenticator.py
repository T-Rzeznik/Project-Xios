import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from functions import reset
import os

load_dotenv('keys.env')

mongo_uri = os.getenv("MONGODB_URI")


def get_database():
    client = MongoClient(mongo_uri)
    return client['ragbot']

def register_user(username, password):
    db = get_database()
    users_collection = db['users']
    if users_collection.find_one({"username": username}):
        st.error("Username already exists. Please choose a different one.")
        return False
    users_collection.insert_one({"username": username, "password": password})
    st.success("User registered successfully!")
    return True

def login_user(username, password):
    db = get_database()
    users_collection = db['users']
    user = users_collection.find_one({"username": username, "password": password})
    if user:
        st.session_state['username'] = username
        st.success("Logged in successfully!")
        return True
    else:
        st.error("Invalid username or password.")
        return False

def logout_user():
    if 'username' in st.session_state:
        del st.session_state['username']
        st.success("Logged out successfully!")

def authenticator(model):
    if 'username' in st.session_state:
        st.sidebar.subheader(f"Welcome, {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()
    else:
        st.sidebar.subheader("Login / Register")
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form(key='login_form'):
                username = st.text_input("Username")
                password = st.text_input("Password", type='password')
                submit_button = st.form_submit_button("Login")
                if submit_button:
                    if login_user(username, password):
                        st.session_state['login_successful'] = True
                        reset(model)
                        st.rerun()
                    else:
                        st.session_state['login_successful'] = False

        with tab2:
            with st.form(key='register_form'):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type='password')
                register_button = st.form_submit_button("Register")
                if register_button:
                    register_user(new_username, new_password)
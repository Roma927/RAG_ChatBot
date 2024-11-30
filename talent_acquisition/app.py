import streamlit as st
import requests
import os

st.title("Talent Acquisition Chatbot")

# Upload CVs
st.subheader("Process CVs")
directory = st.text_input("Enter the directory containing CVs:")
if st.button("Process CVs"):
    st.write("Processing CVs...")
    # Call Django management command (replace with proper API in production)
    os.system(f"python manage.py process_cvs --directory {directory}")
    st.success("CVs processed successfully!")

# Chat with Bot
st.subheader("Chat with the Bot")
query = st.text_input("Enter your query:")
if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/api/chatbot/", json={"query": query})
    st.write(response.json()["answer"])

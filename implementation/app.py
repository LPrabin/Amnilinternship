import streamlit as st


st.title("Model Deployment App")

st.chat_input("You: ", key="input", accept_files= "multiple" , file_types=["jpeg", "png", "jpg"])

if st.button("Submit"):
    st.write("uploading files...")
    
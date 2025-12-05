import time
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Notebook", layout="wide")

st.title("Notebook")

# Sidebar for Notebook Selection
st.sidebar.header("Notebooks")
#have dedicated chat for every notebook

# Fetch notebooks
try:
    notebooks = requests.get(f"{API_URL}/notebooks").json()
except requests.exceptions.ConnectionError:
    st.error("Backend is starting... please wait.")
    time.sleep(5)
    st.rerun()

selected_notebook = st.sidebar.selectbox("Select Notebook", notebooks)

new_notebook_name = st.sidebar.text_input("New Notebook Name")
if st.sidebar.button("Create Notebook"):
    if new_notebook_name:
        requests.post(f"{API_URL}/notebooks", json={"name": new_notebook_name})
        st.rerun()

if selected_notebook:
    if st.sidebar.button("Delete Notebook"):
        requests.delete(f"{API_URL}/notebooks/{selected_notebook}")
        st.rerun()

    st.header(f"Notebook: {selected_notebook}")
    
    tab1, tab2 = st.tabs(["Chat", "Sources"])
    
    with tab2:
        st.subheader("Manage Sources")
        uploaded_file = st.file_uploader("Add Source (PDF/TXT)", type=["pdf", "txt"])
        if uploaded_file:
            if st.button("Upload"):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                with st.spinner("Processing..."):
                    res = requests.post(f"{API_URL}/notebooks/{selected_notebook}/resources", files=files)
                    if res.status_code == 200:
                        st.success("File added!")
                        st.rerun()
                    else:
                        st.error("Failed to upload.")

        st.subheader("Existing Sources")
        resources = requests.get(f"{API_URL}/notebooks/{selected_notebook}/resources").json()
        for res in resources:
            col1, col2 = st.columns([4, 1])
            col1.text(res)
            if col2.button("Delete", key=res):
                requests.delete(f"{API_URL}/notebooks/{selected_notebook}/resources/{res}")
                st.rerun()

    with tab1:
        st.subheader("Chat with your Notebook")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = requests.post(f"{API_URL}/notebooks/{selected_notebook}/query", json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = list(set(data["sources"]))
                    top3docs_data = data.get("top3docs", []) 
                    top3docs = list(top3docs_data)
                    
                    
                    
                    docs_content = "\n\n".join([f"**Source {i+1}:**\n> {doc}" for i, doc in enumerate(top3docs)])
                    full_response = f"{answer}\n\n**Sources:** {', '.join(sources)}\n\n**Retrieved Context:**\n\n{docs_content}"
                    
                    with st.chat_message("assistant"):
                        st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Error getting response.")

else:
    st.info("Select or create a notebook to get started.")

import streamlit as st


st.title("Model Deployment App")

st.write("Upload images to get predictions from the model.")
uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if st.button("Predict"):
    #call the API endpoint to get predictions
    if uploaded_files:
        import requests

        files = [('files', (file.name, file, 'image/jpeg')) for file in uploaded_files]
        response = requests.post("http://localhost:8000/input/", files=files)
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            for file, prediction in zip(uploaded_files, predictions):
                # display image and prediction
                st.image(file)
                st.write(f"File: {file.name} - Prediction: {prediction}")
        else:
            st.error("Error in getting predictions from the server.")
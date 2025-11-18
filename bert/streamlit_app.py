import os
import streamlit as st
import requests


DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(page_title="Nepali Sentiment Classifier", layout="centered")
st.title("Nepali Sentiment Classifier — Streamlit UI")

st.markdown(
    "Use this UI to send text to the FastAPI backend and get sentiment predictions and inference metrics. Input will be truncated to the model's maximum length - 64 tokens."
)

backend_url = st.text_input("FastAPI backend URL", value=DEFAULT_BACKEND)

with st.expander("Example inputs"):
    st.write("- मेरो दिन राम्रो थियो")
    st.write("- मलाई यो कुरा मन परेन")

text = st.text_area("Enter Nepali text to classify", height=180)

# Top row: Predict + Health
cols = st.columns([2, 1])
predict_btn = cols[0].button("Predict")
health_btn = cols[1].button("Health Check")

# Second row: Download model (it can take minutes)
cols2 = st.columns([2, 1])
download_btn = cols2[1].button("Download model on backend")


def call_predict(url: str, text: str):
    url = url.rstrip('/') + '/predict'
    try:
        # The FastAPI endpoint expects a `text` parameter; sending as query params is compatible.
        resp = requests.post(url, params={"text": text}, timeout=30)
        return resp
    except Exception as e:
        raise


def call_download(url: str):
    url = url.rstrip('/') + '/download'
    try:
        # Downloading model may take a long time; give a longer timeout
        resp = requests.post(url, timeout=600)
        return resp
    except Exception as e:
        raise


if health_btn:
    try:
        r = requests.get(backend_url.rstrip('/') + '/', timeout=5)
        if r.ok:
            st.success("Backend reachable")
            try:
                st.json(r.json())
            except Exception:
                st.write(r.text)
        else:
            st.error(f"Backend returned status {r.status_code}")
    except Exception as e:
        st.error(f"Health check failed: {e}")


if predict_btn:
    if not text or not text.strip():
        st.warning("Please enter some text to classify.")
    else:
        try:
            with st.spinner("Calling backend for prediction..."):
                resp = call_predict(backend_url, text)

            if resp is None:
                st.error("No response received from backend.")
            elif resp.status_code != 200:
                st.error(f"Backend error {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                st.success(data.get("sentiment", "Prediction returned"))
                st.subheader("Prediction Details")
                st.write("**Input:**")
                st.write(data.get("input_text", text))
                st.write("**Label (numeric):**", data.get("prediction_label"))
                st.write("**Sentiment:**", data.get("sentiment"))

                if "metrics" in data:
                    st.subheader("Inference Metrics")
                    st.json(data["metrics"])

                st.subheader("Full Response")
                st.json(data)

        except Exception as e:
            st.error(f"Request failed: {e}")


if download_btn:
    try:
        with st.spinner("Requesting backend to download model (this can take minutes)..."):
            resp = call_download(backend_url)

        if resp is None:
            st.error("No response from backend.")
        elif resp.status_code != 200:
            st.error(f"Backend error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            if data.get("status") == "ok":
                st.success("Model downloaded and loaded on backend")
                st.write("Snapshot path:", data.get("snapshot_path"))
                st.write("Time (s):", data.get("time_s"))
            else:
                st.error("Backend returned an error: " + str(data))

    except Exception as e:
        st.error(f"Download request failed: {e}")


st.markdown("---")
st.markdown(
    "Run the FastAPI server (example): `uvicorn main:app --host 0.0.0.0 --port 8000`\n\nThen run Streamlit:\n`streamlit run streamlit_app.py`"
)


import os
import time
import streamlit as st
import requests


DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(page_title="Nepali Sentiment Classifier", layout="centered")
st.title("Nepali Sentiment Classifier — Streamlit UI")

st.markdown(
    "Use this UI to send text to the FastAPI backend and get sentiment predictions and inference metrics. Input will be truncated to the model's maximum length - 64 tokens now 128."
)

backend_url = st.text_input("FastAPI backend URL", value=DEFAULT_BACKEND)

with st.expander("Example inputs"):
    st.write("- मेरो दिन राम्रो थियो")
    st.write("- मलाई यो कुरा मन परेन")


def check_status(url: str):
    url = url.rstrip('/') + '/status'
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        return {"loaded": False}
    return {"loaded": False}


# Query status on load
status = check_status(backend_url)
model_loaded = bool(status.get("loaded"))
snapshot_path = status.get("snapshot_path")

# persist state across reruns
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = model_loaded
if "snapshot_path" not in st.session_state:
    st.session_state["snapshot_path"] = snapshot_path

# Top row: Predict + Health
cols = st.columns([2, 1])
health_btn = cols[1].button("Health Check")



if st.session_state.get("model_loaded"):
    st.success("Model is available on backend")
    if st.session_state.get("snapshot_path"):
        st.write("Snapshot:", st.session_state.get("snapshot_path"))
else:
    st.warning("Model not available yet. Use the button to download or wait until it's loaded.")


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


if model_loaded:
    text = st.text_area("Enter Nepali text to classify", height=180)
    cols_pred = st.columns([2, 1])
    predict_btn = cols_pred[0].button("Predict")

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
else:
    st.info("Prediction is disabled until the model is available on the backend.")



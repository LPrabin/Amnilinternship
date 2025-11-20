import time
import psutil
import torch
import logging
import os
from transformers import AutoTokenizer 
from optimum.onnxruntime import ORTModelForSequenceClassification
from fastapi import FastAPI
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_ID = 'AlgoAlchemist/nepalisentimentbert'
# Hugging Face default cache directory
HF_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub')

# Global variables for model and tokenizer
ort_model = None
tokenizer = None
LAST_SNAPSHOT_PATH = None

def is_model_downloaded_locally(model_name, cache_dir):
    """
    Checks if the model snapshot is already fully downloaded in the HF cache directory.
    This prevents logging metrics during the actual network download phase.
    """
    snapshot_path = snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True, revision="main")

    return snapshot_path

@app.on_event("startup")
def load_model_and_tokenizer():
    global ort_model, tokenizer
    
    is_downloaded = False
    global LAST_SNAPSHOT_PATH
    try:
        # Check if the model files are local only. This raises an error if not found locally.
        snapshot_path = is_model_downloaded_locally(MODEL_ID, HF_CACHE_DIR)
        LAST_SNAPSHOT_PATH = snapshot_path
        is_downloaded = True
        logger.info("Model snapshot found locally. Proceeding with logging model loading metrics.")
    except Exception as e:
        # This branch runs if the model needs to be downloaded first.
        logger.info("Model snapshot not found locally. Downloading model first. Metrics will be logged *after* the initial download phase is complete.")

    # --- Start the actual loading process (which might include a download if not cached) ---
    
    start_time = time.time()
    process = psutil.Process(os.getpid()) # Use getpid() for the current process

    # Capture metrics BEFORE load attempt
    memory_before = process.memory_info().rss / (1024 * 1024)
    cpu_before = process.cpu_percent(interval=None)
    gpu_before = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    try:
        # Prefer loading from cache when possible; transformers will use the HF cache dir.
        # If we have a known snapshot path, prefer loading from that local path.
        if LAST_SNAPSHOT_PATH:
            tokenizer = AutoTokenizer.from_pretrained(LAST_SNAPSHOT_PATH, local_files_only=True)
            ort_model = ORTModelForSequenceClassification.from_pretrained(LAST_SNAPSHOT_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID)
        logger.info("Model and tokenizer load sequence complete.")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model {MODEL_ID}") from e
        
    end_time = time.time()
    
    # Only log the detailed metrics block IF the model was found locally *initially*
    if is_downloaded:
        memory_after = process.memory_info().rss / (1024 * 1024)
        cpu_after = process.cpu_percent(interval=0) 
        gpu_after = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

        model_loading_time = (end_time - start_time) * 1000

        metrics = {
            "memory_before_mb": round(memory_before, 2),
            "cpu_before_percent": cpu_before,
            "gpu_before_mb": round(gpu_before, 2),
            "memory_after_mb": round(memory_after, 2),
            "cpu_after_percent": cpu_after,
            "gpu_after_mb": round(gpu_after, 2),
            "model_loading_time_ms": round(model_loading_time, 2)
        }
        logger.info(f"Model loading performance metrics (loaded from cache): {metrics}")
    else:
        logger.info(f"Model download just completed. Total time to download and load: {(end_time - start_time):.2f} seconds. Metrics not captured during download phase.")
    
MAX_LENGTH = 128
@app.get("/")
def read_root():
    return {"message": "Nepali Sentiment Classifier API is running."}

@app.post("/predict")
def predict_sentiment(text: str):
    """Predicts the sentiment of the input text."""
    if ort_model is None or tokenizer is None:
        return {"error": "Model not loaded. Server is starting up."}
    start_time = time.time()
    process = psutil.Process()

    memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    cpu_before = process.cpu_percent(interval=None)
    gpu_before = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    
    
    inputs_onnx = {k: v.cpu().numpy() for k, v in inputs.items()}
    outputs = ort_model(**inputs_onnx)
        

    logits = outputs.get('logits', outputs[0]) if isinstance(outputs, dict) else outputs[0]
    
    predictions = torch.argmax(torch.from_numpy(logits), dim=1).item()
    sentiment = "Positive (2)" if predictions == 2 else "Neutral (1)" if predictions == 1 else "Negative (0)"

    

    end_time = time.time()
    memory_after = process.memory_info().rss / (1024 * 1024)
    cpu_after = process.cpu_percent(interval=None)
    gpu_after = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    inference_time = (end_time - start_time) * 1000  # in milliseconds
    metrics = {
        "memory_before_mb": round(memory_before, 2),
        "cpu_before_percent": cpu_before,
        "gpu_before_mb": round(gpu_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "cpu_after_percent": cpu_after,
        "gpu_after_mb": round(gpu_after, 2),
        "inference_time_ms": round(inference_time, 2)
    }
    logger.info(f"Inference metrics: {metrics}")
    return {"input_text": text, "sentiment": sentiment, "prediction_label": predictions, "metrics": metrics}


@app.post("/download")
def download_model():
   
    global ort_model, tokenizer
    global LAST_SNAPSHOT_PATH
    try:
        start = time.time()
        logger.info(f"Starting model download for {MODEL_ID} into {HF_CACHE_DIR}")
        # snapshot_download will return the path to the downloaded repo snapshot
        snapshot_path = snapshot_download(repo_id=MODEL_ID, cache_dir=HF_CACHE_DIR, revision="main")
        LAST_SNAPSHOT_PATH = snapshot_path

        # Load locally from the snapshot path to ensure no network calls on future loads
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        ort_model = ORTModelForSequenceClassification.from_pretrained(snapshot_path)

        elapsed = time.time() - start
        logger.info(f"Model downloaded and loaded from {snapshot_path} in {elapsed:.2f}s")
        return JSONResponse({"status": "ok", "snapshot_path": snapshot_path, "time_s": round(elapsed, 2)})
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/status")
def status():
    """Return whether the model is loaded and the last snapshot path (if any)."""
    return {"loaded": ort_model is not None and tokenizer is not None, "snapshot_path": LAST_SNAPSHOT_PATH}
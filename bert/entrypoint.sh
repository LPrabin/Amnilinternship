#!/bin/bash

# Ensure you are in the correct directory if entrypoint.sh is executed from somewhere else
cd /app

echo "Starting FastAPI..."
# Uvicorn needs the module 'api' to be in the current PYTHONPATH
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run streamlit_app.py --server.port 8001 --server.address 0.0.0.0
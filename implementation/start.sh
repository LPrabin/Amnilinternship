#!/bin/bash


echo "Starting Modle Implementation Script..."

uvicorn implementation:app --host 0.0.0.0 --port 8000 &


echo "starting streamlit"

streamlit run app.py --server.port 8501 --server.address=0.0.0.0

wait -n

exit $?

# bert nepali sentiment classifier
Initial run takes time because the model from Hugging Face needs to be downloaded, size is about 500mb
downloads automatically on startup of uvcorn

## for docker run
[may take time to build Docker](Docker file only copies necessary files so can use bert directory)
cd bert 
docker build -t bert .
docker run  --name container -p 8000:8000 -p 8001:8001 bert

## for local virtual env run 

cd bert 
create venv -> activate
pip install requirements.txt
bash entrypoint.sh

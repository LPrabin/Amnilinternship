
# bert nepali sentiment calssifier
initial run takes time because model from hugging face needs to be download , size is about 500mb
downloads automatically on start up of uvcorn

## for docker run
[may take time to build docker ]
cd bert 
docker build -t bert .
docker run  --name container -p 8000:8000 -p 8001:8001 bert

### for local virtual env run 

cd bert 
create venv -> activate
pip install requirements.txt
bash entrypoint.sh

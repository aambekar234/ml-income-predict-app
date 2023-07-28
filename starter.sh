#!/bin/bash
pip install --upgrade pip
pip3 install -r requirements.txt
mkdir keys
ls -al

cat /etc/secrets/sa.json >> keys/sa-key.json
cat keys/sa-key.json
dvc remote default

# echo "******** Now doing dvc pull ************"
# dvc pull
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app
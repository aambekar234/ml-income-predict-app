#!/bin/bash
pip install --upgrade pip
pip3 install -r requirements.txt
mkdir keys
ls -al

cat /etc/secrets/sa.json >> keys/sa-key.json
echo keys/sa-key.json

dvc pull
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app
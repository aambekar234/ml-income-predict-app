#!/bin/bash
pip3 install -r requirements.txt
mkdir keys
echo "${sakey}" >> keys/sa-key.json
dvc pull
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app
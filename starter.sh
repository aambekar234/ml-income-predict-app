#!/bin/bash
mkdir keys
echo "${sakey}" >> keys/sa-key.json
dvc pull
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app
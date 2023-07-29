#!/bin/bash

ls -la

# echo "******** Now doing dvc pull ************"
conda init bash
source ~/.bashrc
conda activate base-ml-py3.8
dvc remote default
dvc pull
gunicorn -w 3 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3100 main:app
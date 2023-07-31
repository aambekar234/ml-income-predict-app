#!/bin/bash

ls -la
conda init bash
source ~/.bashrc
conda activate base-ml-py3.8
# echo ******** Now doing dvc pull ************
dvc pull
# echo ******** Starting the Service ************
gunicorn -w 3 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3100 main:app
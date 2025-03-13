#!/bin/bash

# export jupyter notebook on port 8888
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' &

# Start the Flask application (listening on port 5000)
python app.py

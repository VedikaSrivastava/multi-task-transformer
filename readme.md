# Multi-Task Sentence Transformer

This repository contains an implementation of a multi-task learning architecture built on top of a sentence transformer model. The project demonstrates how to extend a transformer-based encoder to simultaneously support:

- **Task A: Sentence Classification**
- **Task B: Named Entity Recognition (NER)**

The demo includes:
- A Flask web application for interacting with the model
- A Jupyter Notebook (`code.ipynb`) for reviewing and running training/experimental code


## *[Writeup](writeup.md) file provides detailed explanations regarding architecture choices, training strategies, and additional experiment details*


## Project Structure

```
ROOT/
├── templates/
|    └── index.html       # html template for the Flask web interface
├── app.py                # flask application serving the multi-task model endpoints
├── best_model.pth        # pre-trained model weights
├── code.ipynb            # jupyter notebook for training/review purposes
├── Dockerfile            # Dockerfile to build the container
├── requirements.txt      # list of required python packages
├── run.sh                # docker commands for running scripts
├── writeup.md            # detailed strategy of model design
└── readme.md             # this file

```

## Instruction to build and run docker image

To simplify evaluation and testing, the entire application is packaged in a Docker container

### 1. Build the Docker Image

Run the following command in project directory to build the Docker image:

```
docker build -t multi-task-model .
```
> *NOTE: Alternatively clone pre-bild dcoker image from using the command `docker pull vedikasrivastavr/multi-task-model`*


### 2. Run the Docker Container
After building the image, run the container with the following command:

```
docker run -p 5000:5000 -p 8888:8888 multi-task-model
```


This command maps:
- Port 5000: For the Flask web application
    - Flask App: Open your browser and navigate to http://localhost:5000 to interact with the multi-task model (try sentence classification, NER, or embedding tasks)
- Port 8888: For the Jupyter Notebook(code.ipynb) to review and run training code
    - Jupyter Notebook: Open your browser and navigate to http://localhost:8888 to view and run the code.ipynb
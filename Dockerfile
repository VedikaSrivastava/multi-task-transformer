# python base image
FROM python:3.12.8

# working directory in container
WORKDIR /app

# Copy files
COPY requirements.txt ./
COPY app.py ./
COPY templates ./templates
COPY best_model.pth ./
COPY code.ipynb ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter

# Flask port
EXPOSE 5000 

# Jupyter port
EXPOSE 8888

# Copy the startup script and make it executable
COPY run.sh ./
RUN chmod +x run.sh

# Run the startup script
CMD ["./run.sh"]

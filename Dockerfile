FROM python:3.10-slim

# Basic dependencies
RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y build-essential ffmpeg libsm6 libxext6 git nano jq cmake libzbar0 && \
pip install --upgrade pip

# Set working directory 
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Set git config to allow unsafe directory
RUN git config --global --add safe.directory /app

# run bash
CMD ["bash"]

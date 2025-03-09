FROM python:3.10-slim

# Basic dependencies
RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y build-essential ffmpeg libsm6 libxext6 git nano jq cmake libzbar0 && \
pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# run bash
CMD ["bash"]

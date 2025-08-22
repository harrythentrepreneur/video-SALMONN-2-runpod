# RunPod Dockerfile for video-SALMONN 2
# Using RunPod's base image for better compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK and additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    opencv-python-headless \
    imageio \
    imageio-ffmpeg \
    python-dotenv

# Copy the application code
COPY . /workspace/

# Create model directory
RUN mkdir -p /workspace/models

# Environment variables for model paths
ENV MODEL_PATH=/workspace/models/video-SALMONN-2
ENV MODEL_BASE=/workspace/models/video-SALMONN-2
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Download model weights (optional - can be done at runtime)
# Uncomment the following lines to download model during build
# RUN huggingface-cli download tsinghua-ee/video-SALMONN-2 \
#     --local-dir /workspace/models/video-SALMONN-2

# Make handler executable
RUN chmod +x runpod_serverless.py

# Set the handler as the default command
# Using the production-ready serverless handler
CMD ["python", "-u", "runpod_serverless.py"]
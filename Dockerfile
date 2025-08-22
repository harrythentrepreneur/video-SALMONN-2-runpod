# RunPod Dockerfile for video-SALMONN 2
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# First upgrade pip to avoid issues
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements_runpod.txt /workspace/requirements_runpod.txt
RUN pip install --no-cache-dir -r requirements_runpod.txt || \
    (echo "Some packages failed, installing core only..." && \
     pip install --no-cache-dir torch torchvision transformers runpod)

# Copy the application code
COPY . /workspace/

# Create model directory
RUN mkdir -p /workspace/models

# Environment variables for model paths
ENV MODEL_PATH=/workspace/models/video-SALMONN-2
ENV MODEL_BASE=/workspace/models/video-SALMONN-2
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}
ENV CUDA_VISIBLE_DEVICES=0

# Download model weights (optional - can be done at runtime)
# Uncomment the following lines to download model during build
# RUN huggingface-cli download tsinghua-ee/video-SALMONN-2 \
#     --local-dir /workspace/models/video-SALMONN-2

# Make handlers executable
RUN chmod +x runpod_minimal.py runpod_serverless.py || true

# Set the handler as the default command
# Using video-SALMONN-2 with PEFT for advanced video understanding
CMD ["python", "-u", "runpod_salmonn.py"]
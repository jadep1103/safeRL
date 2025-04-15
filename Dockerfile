# Use official Python 3.10 base image
FROM python:3.10-slim

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install --no-cache-dir \
    safety-gymnasium==1.0.0 \
    torch>=2.3 \
    tensorboard \
    swig \
    stable-baselines3 \
    requests \
    wandb
RUN pip install --no-cache-dir "gymnasium[box2d]"

# Default command
CMD ["/bin/bash"]

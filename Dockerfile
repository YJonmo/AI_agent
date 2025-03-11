# Use NVIDIA's official PyTorch image as the base
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variables to enable GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the working directory
WORKDIR /app

# Copy your requirements file (if applicable)
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of your project files
COPY . /app/

# RUN ln -s /usr/bin/python3 /usr/bin/python

# Expose the port that the application listens on.
EXPOSE 8080

# Command to run the application
# CMD ["python", "src.models.py"]
CMD ["uvicorn", "src.agent:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# in command line run
# sudo docker build -t ai_agent .
# sudo docker run --gpus all -p 8080:8080 ai_agent 
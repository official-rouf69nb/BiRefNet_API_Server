# Use the official PyTorch image with CUDA support as base.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Install system dependencies required by PyTorch and Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt /requirements.txt

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code into the container.
COPY app.py /app.py

# Expose port 8501.
EXPOSE 8501

# Command to run the FastAPI server with Uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]

#docker run --gpus all -p 8000:8000 fastapi-cuda-app

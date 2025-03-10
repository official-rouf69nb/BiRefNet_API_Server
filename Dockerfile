# Use the official Python 3.10 slim image as base.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies required by PyTorch and Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container.
COPY requirements.txt /app/requirements.txt

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code into the container.
COPY app.py /app/app.py

# Expose port 8501.
EXPOSE 8501

# Command to run the FastAPI server with Uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]

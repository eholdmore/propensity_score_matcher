# Use a slim Python image for efficiency
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies [cite: 37]
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and the data directory
COPY . .

# Streamlit default port
EXPOSE 8501

# Run the app and ensure it listens on all interfaces for Cloud Run/K8s compatibility
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.10.17-slim

# Set working directory
WORKDIR /app

# Install git, git-lfs, and required system libraries
RUN apt-get update && \
    apt-get install -y git git-lfs libgomp1 && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with aggressive retry handling
# Skip pip upgrade to avoid initial timeout, use existing pip
RUN pip install --no-cache-dir --retries 20 --timeout 300 -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Expose port
EXPOSE 5003

# Run with gunicorn
CMD ["gunicorn", "app:app", "-w", "2", "-b", "0.0.0.0:5003", "--timeout", "300", "--preload", "--log-level", "info"]



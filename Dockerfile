# BD News RAG Chatbot Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHROMA_TELEMETRY=false
ENV ANONYMIZED_TELEMETRY=false
ENV CHROMA_SERVER_NOFILE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (with retries + HTTP/1.1)
RUN for i in 1 2 3; do \
      curl -fsSL --http1.1 https://ollama.com/install.sh | sh && break || \
      (echo "Retrying Ollama install ($i/3)..." && sleep 10); \
    done

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs src

# Expose ports
EXPOSE 8501 11434

# Create startup script with proper Streamlit app path
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Ollama server..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
echo "Waiting for Ollama server to start..."\n\
sleep 10\n\
\n\
# Test Ollama connection\n\
max_attempts=30\n\
attempt=1\n\
while [ $attempt -le $max_attempts ]; do\n\
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then\n\
        echo "Ollama server is ready!"\n\
        break\n\
    else\n\
        echo "Waiting for Ollama server... (attempt $attempt/$max_attempts)"\n\
        sleep 2\n\
        attempt=$((attempt + 1))\n\
    fi\n\
done\n\
\n\
if [ $attempt -gt $max_attempts ]; then\n\
    echo "Error: Ollama server failed to start after $max_attempts attempts"\n\
    exit 1\n\
fi\n\
\n\
echo "Pulling Llama 3.2 model..."\n\
ollama pull llama3.2 || echo "Warning: Could not pull model. Will try to use existing model."\n\
\n\
echo "Starting Streamlit application..."\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check for both Ollama and Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && curl -f http://localhost:11434/api/tags || exit 1

# Default command
CMD ["/app/start.sh"]

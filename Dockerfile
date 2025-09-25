FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy app files and entrypoint script
COPY . .

RUN sudo apt update && sudo apt install curl

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE 5000

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]



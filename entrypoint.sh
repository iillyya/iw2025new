#!/bin/bash
set -e

# Wait for Qdrant
echo "⏳ Waiting for Qdrant..."
for i in $(seq 1 30); do
    curl -s $QDRANT_URL/collections && break || sleep 2
done

# Initialize collection
python init_qdrant.py

# Start Flask
python chat_app.py

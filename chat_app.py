from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fireworks.client import Fireworks
import os

# Flask app
app = Flask(__name__)

# Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "mpea"

# Embedding model (same as used for your dataset)
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Fireworks client
api_key = os.getenv("FIREWORKS_API_KEY")
if api_key is None:
    raise Exception("API key is not provided. Pass it to the FIREWORKS_API_KEY parameter.")
fw = Fireworks(api_key=api_key)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # 1. Embed query
    query_vector = embedder.encode(user_query).tolist()

    # 2. Retrieve relevant vectors from Qdrant
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    # 3. Build context from retrieved payloads
    context = "\n".join([str(r.payload) for r in search_result])

    # 4. Ask Fireworks LLM
    response = fw.chat.completions.create(
        model="accounts/fireworks/models/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a materials science assistant."},
            {"role": "user", "content": f"Question: {user_query}\n\nContext (this is information from MPEA datset. it may be useful information but do not rely completely on it): {context}"}
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    return jsonify({
        "query": user_query,
        "answer": answer,
        "context": context
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

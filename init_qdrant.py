# init_qdrant.py
import pickle
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
import os

# Connect to Qdrant (use docker-compose service name)
qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
client = QdrantClient(url=qdrant_url)

# Load embeddings
with open("mpea_embeddings.pkl", "rb") as f:
    embeds = pickle.load(f)

# Load metadata (adjust if your file is named differently)
df = pd.read_csv("MPEA.csv")

# Create or recreate collection
client.recreate_collection(
    collection_name="mpea",
    vectors_config=models.VectorParams(
        size=embeds.shape[1],
        distance=models.Distance.COSINE
    ),
)

# Upload points
points = [
    PointStruct(
        id=i,
        vector=embeds[i].tolist(),
        payload={"formula": df.loc[i, "FORMULA"]}
    )
    for i in range(len(embeds))
]

client.upsert(collection_name="mpea", points=points)
print("✅ Qdrant collection 'mpea' created and populated")

import pickle
from sentence_transformers import SentenceTransformer

# Step 1: Load the chunks from chunks.pkl
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Step 2: Generate embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

# Step 3: Save the embeddings
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Embeddings generated and saved to embeddings.pkl")

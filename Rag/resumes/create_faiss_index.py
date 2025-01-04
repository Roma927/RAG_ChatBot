import pickle
import numpy as np
import faiss

# Step 1: Load the embeddings from embeddings.pkl
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Step 2: Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Step 3: Create a FAISS index
dimension = embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings_array)  # Add embeddings to the index

# Step 4: Save the FAISS index to disk
faiss.write_index(index, 'resumes_index.faiss')

print("FAISS index created and saved to resumes_index.faiss")

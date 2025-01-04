import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Step 1: Load the FAISS index
index = faiss.read_index('resumes_index.faiss')

# Step 2: Load the chunks from chunks.pkl
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Step 3: Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Define the search function
def search_resumes(query, k=3):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array([query_embedding])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Prepare the results
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "rank": i + 1,
            "distance": float(distances[0][i]),
            "text": chunks[idx].page_content,
            "source": chunks[idx].metadata['source']
        }
        results.append(result)

    return results

# Step 5: Define the chatbot interface
def chatbot_interface(query):
    results = search_resumes(query)
    output = ""
    for result in results:
        output += f"Rank: {result['rank']}\n"
        output += f"Source: {result['source']}\n"
        output += f"Text: {result['text']}\n"
        output += "-" * 50 + "\n"
    return output

# Step 6: Launch the Gradio app
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Resume Screening Chatbot",
    description="Ask questions about candidates' resumes."
)
iface.launch()

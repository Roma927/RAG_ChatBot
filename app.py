# Install Required Packages
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import os
import faiss
import numpy as np
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# 1. Text Extraction from CVs (Images and PDFs)
def extract_text_from_cv(cv_path):
    """Extract text from CV (image or PDF)."""
    if cv_path.endswith('.pdf'):
        raise NotImplementedError("Add logic for PDF processing (e.g., PyMuPDF)")
    else:
        image = cv2.imread(cv_path)
        return pytesseract.image_to_string(Image.fromarray(image))

# 2. Text Chunking
def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 3. Generate Embeddings
def generate_embeddings(text_chunks):
    """Generate embeddings for text chunks using SentenceTransformer."""
    embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    return [embedder.embed(chunk) for chunk in text_chunks]

# 4. Initialize FAISS Index
def initialize_faiss_index(dimension):
    """Initialize a FAISS index."""
    index = faiss.IndexFlatL2(dimension)  # L2 similarity search
    return index

# 5. Store Embeddings with Metadata
metadata_store = {}

def store_embeddings_with_metadata(faiss_index, embeddings, metadata):
    """Store embeddings and metadata in FAISS."""
    embeddings_np = np.array(embeddings).astype("float32")
    index_id = len(metadata_store)
    for embedding, meta in zip(embeddings_np, metadata):
        faiss_index.add(np.expand_dims(embedding, axis=0))
        metadata_store[index_id] = meta
        index_id += 1
    return faiss_index

# 6. Process Single CV
def process_cv(cv_path, faiss_index, embedding_dim=384):
    """Process a single CV."""
    # Extract text
    text = extract_text_from_cv(cv_path)

    # Chunk text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store embeddings with metadata
    metadata = [{"filename": os.path.basename(cv_path), "chunk": i} for i in range(len(chunks))]
    faiss_index = store_embeddings_with_metadata(faiss_index, embeddings, metadata)

    return faiss_index

# 7. Process All CVs in Directory
def process_all_cvs(cv_directory, faiss_index, embedding_dim=384):
    """Process all CVs in a directory."""
    for filename in os.listdir(cv_directory):
        if filename.endswith((".jpg", ".png", ".pdf")):  # Add supported file types
            cv_path = os.path.join(cv_directory, filename)
            faiss_index = process_cv(cv_path, faiss_index, embedding_dim)
    return faiss_index

# 8. Query Candidates
def query_candidates(faiss_index, user_query, embedding_dim=384, k=5):
    """Search for top K candidates."""
    query_embedding = generate_embeddings([user_query])[0]
    query_embedding_np = np.array(query_embedding).reshape(1, embedding_dim).astype('float32')
    distances, indices = faiss_index.search(query_embedding_np, k)
    results = [{"candidate": metadata_store[idx], "distance": dist} for idx, dist in zip(indices[0], distances[0])]
    return results

# 9. RAG Integration
def setup_rag_model(faiss_index):
    """Setup RAG model with FAISS and LangChain."""
    vector_store = FAISS(faiss_index, embedding_dim=384)
    retriever = vector_store.as_retriever()
    llm = OpenAI(model_name="gpt-4")  # Replace with a suitable OpenAI model
    return RetrievalQA(llm=llm, retriever=retriever)

def ask_bot(rag_model, query):
    """Query the RAG model."""
    return rag_model.run(query)

# Initialize FAISS Index and RAG Model
embedding_dim = 384
faiss_index = initialize_faiss_index(embedding_dim)
rag_model = None

@app.on_event("startup")
async def startup():
    # Startup logic here
    pass

@app.on_event("shutdown")
async def shutdown():
    # Shutdown logic here
    pass

# API Endpoints
@app.post("/upload_cv/")
async def upload_cv(file: UploadFile):
    """Endpoint to upload and process a CV."""
    content = await file.read()
    cv_path = f"/tmp/{file.filename}"
    with open(cv_path, "wb") as f:
        f.write(content)

    global faiss_index
    faiss_index = process_cv(cv_path, faiss_index, embedding_dim)
    return {"message": "CV processed and stored in FAISS index."}

@app.get("/query/")
async def query_candidates_endpoint(query: str):
    """Endpoint to query similar candidates."""
    results = query_candidates(faiss_index, query, embedding_dim, k=5)
    return {"results": results}

@app.get("/chat/")
async def chat_with_bot(query: str):
    """Endpoint to chat with the bot."""
    response = ask_bot(rag_model, query)
    return {"response": response}

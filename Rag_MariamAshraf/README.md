# Resume Screening Chatbot with Generative AI

This project is a **Resume Screening Chatbot** that uses Generative AI and vector databases to help recruiters find the best candidates based on their resumes. The chatbot processes resumes, generates embeddings, stores them in a vector database (FAISS), and allows users to query the resumes using natural language.

---

## **Features**
- **Resume Processing**: Extracts text from PDF resumes and splits it into manageable chunks.
- **Embedding Generation**: Converts text chunks into vector embeddings using a pre-trained model (`all-MiniLM-L6-v2`).
- **Vector Database**: Stores embeddings in a FAISS index for fast similarity search.
- **Chatbot Interface**: Provides a user-friendly interface (Gradio) to query resumes using natural language.

---

## **How It Works**
1. **Resume Processing**:
   - The script `process_resumes.py` extracts text from PDF resumes and splits it into chunks.
   - The chunks are saved in `chunks.pkl`.

2. **Embedding Generation**:
   - The script `generate_embeddings.py` generates embeddings for the text chunks using a pre-trained model.
   - The embeddings are saved in `embeddings.pkl`.

3. **FAISS Index Creation**:
   - The script `create_faiss_index.py` creates a FAISS index from the embeddings and saves it to `resumes_index.faiss`.

4. **Chatbot Interface**:
   - The script `chatbot.py` provides a Gradio interface to query the resumes.
   - Users can ask questions like:
     - "Find candidates with experience in Python."
     - "Show me resumes with machine learning skills."

---

## **Installation**

### **1. Install Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **2. Prepare Resumes**
- Place your PDF resumes in the `resumes` directory.

### **3. Run the Scripts**
Run the scripts in the following order:

1. **Process Resumes**:
   ```bash
   python process_resumes.py
   ```
   - This extracts text from resumes and saves the chunks in `chunks.pkl`.

2. **Generate Embeddings**:
   ```bash
   python generate_embeddings.py
   ```
   - This generates embeddings for the text chunks and saves them in `embeddings.pkl`.

3. **Create FAISS Index**:
   ```bash
   python create_faiss_index.py
   ```
   - This creates a FAISS index from the embeddings and saves it to `resumes_index.faiss`.

4. **Run the Chatbot**:
   ```bash
   python chatbot.py
   ```
   - This launches the chatbot interface.

### **4. Access the Chatbot**
- Open your browser and go to `http://127.0.0.1:7860`.
- Start querying the resumes using natural language.

---

## **Requirements**
- Python 3.8+
- Libraries:
  - `pdfminer.six`: For extracting text from PDFs.
  - `langchain`: For text splitting and document handling.
  - `sentence-transformers`: For generating embeddings.
  - `faiss-cpu`: For creating and querying the vector index.
  - `gradio`: For building the chatbot interface.

---

## **File Structure**
```
/resume-screening-chatbot/
├── process_resumes.py       
├── generate_embeddings.py   
├── create_faiss_index.py  
├── chatbot.py               
├── resumes/                 
├── chunks.pkl               
├── embeddings.pkl           
├── resumes_index.faiss      
├── README.md                
├── requirements.txt         
```

---

## **Example Queries**
Here are some example queries you can try in the chatbot:
- "Find candidates with experience in Python."
- "Show me resumes with machine learning skills."
- "Who has worked as a data scientist?"

---

## **Acknowledgments**
- [LangChain](https://langchain.com/) for text splitting and document handling.
- [Sentence Transformers](https://www.sbert.net/) for generating embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) for vector indexing and similarity search.
- [Gradio](https://gradio.app/) for building the chatbot interface.


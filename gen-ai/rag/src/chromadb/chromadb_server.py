from fastapi import FastAPI, UploadFile
from chromadb import Client, Settings
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize ChromaDB client with persistence
client = Client(Settings(persist_directory="/mnt/data/ChromaDb"))

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ingest text data into ChromaDB
@app.post("/ingest")
async def ingest_data(texts: list):
    embeddings = model.encode(texts)
    client.add_texts(collection="documents", texts=texts, embeddings=embeddings)
    return {"status": "success", "message": "Data ingested"}

# Ingest CSV file into ChromaDB
@app.post("/ingest-csv")
async def ingest_csv(file: UploadFile):
    df = pd.read_csv(file.file)
    texts = df['text_column'].tolist()  # Replace 'text_column' with actual column name
    await ingest_data(texts)
    return {"status": "success", "message": "CSV ingested"}

# Helper function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Ingest a PDF file into ChromaDB
@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile):
    file_path = f"/mnt/data/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    extracted_text = extract_text_from_pdf(file_path)
    await ingest_data([extracted_text])
    
    return {"status": "success", "message": "PDF ingested"}

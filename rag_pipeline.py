import uuid
from sqlalchemy import text
from db_connector import SessionLocal
import re
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = 'vectorstore/db_faiss'

def extract_hc_id(filename):
    """Extrae el identificador HC-XXXXX del nombre del archivo."""
    match = re.search(r'HC-\d+', filename)
    return match.group(0) if match else "HC-UNKNOWN"

def ingest_document(file_path):
    filename = os.path.basename(file_path)
    hc_id = extract_hc_id(filename) # Extraemos el ID clínico

    loader = PyPDFLoader(file_path) if file_path.endswith('.pdf') else TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Generar ID único para el índice
    index_id = str(uuid.uuid4())
    vectorstore.save_local(f"{DB_FAISS_PATH}/{index_id}")
    
    # Guardar en PostgreSQL
    db = SessionLocal()
    try:
        query = text("""
            INSERT INTO index_rag_pdf (idex_rag, nombre_pdf, hc_id) 
            VALUES (:id, :nom, :hc)
        """)
        db.execute(query, {"id": index_id, "nom": filename, "hc": hc_id})
        db.commit()

    finally:
        db.close()
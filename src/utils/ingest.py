# src/ingest.py

import os
import glob
from typing import List
from tqdm import tqdm # For progress bars

# LangChain components for ingestion
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_DIR = "./data/clean"  # Directory containing processed .txt files
VECTORSTORE_DIR = "./vectorstore/faiss_index"  # Directory to save FAISS index
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# --- 1. Load Documents ---
def load_documents(data_dir: str) -> List:
    """
    Loads text documents from the specified directory.
    Assumes documents are .txt files.
    """
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not txt_files:
        print(f"⚠️ No .txt files found in {data_dir}. Please run your parsers first.")
        return documents

    print(f"📁 Loading {len(txt_files)} documents from {data_dir}...")
    for file_path in tqdm(txt_files, desc="Loading Files"):
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    print(f"✅ Loaded {len(documents)} document objects.")
    return documents

# --- 2. Split Documents into Chunks ---
def split_documents(documents: List, chunk_size: int = 800, chunk_overlap: int = 100) -> List:
    """
    Splits documents into smaller chunks for better retrieval.
    """
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""] # Try to split on paragraphs, then lines, etc.
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} text chunks.")
    return texts

# --- 3. Create Embeddings ---
def create_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """
    Initializes the HuggingFace embeddings model.
    Consider 'BAAI/bge-small-en-v1.5' for potentially better quality.
    """
    print(f"🧠 Loading embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("✅ Embedding model loaded.")
    return embeddings

# --- 4. Create and Save FAISS Vectorstore ---
def create_and_save_vectorstore(texts: List, embeddings, vectorstore_dir: str):
    """
    Creates a FAISS vectorstore from texts and embeddings, then saves it.
    """
    print("🔍 Creating FAISS vectorstore (this may take a few minutes)...")
    try:
        # Create the vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Save the vectorstore locally
        vectorstore.save_local(vectorstore_dir)
        print(f"✅ FAISS vectorstore created and saved to {vectorstore_dir}")
    except Exception as e:
        print(f"❌ Error creating or saving vectorstore: {e}")

# --- Main Ingestion Process ---
def main():
    """Main function to run the ingestion pipeline."""
    print("--- Starting ZimLaw Ingestion Pipeline ---")
    
    # 1. Load
    documents = load_documents(DATA_DIR)
    if not documents:
        return

    # 2. Split
    texts = split_documents(documents)

    # 3. Embed
    embeddings = create_embeddings()

    # 4. Store
    create_and_save_vectorstore(texts, embeddings, VECTORSTORE_DIR)
    
    print("--- Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    main()
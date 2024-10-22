import os
import json
import time
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants and Configuration
PERSIST_DIRECTORY = "example_data"
EMBEDDING_MODEL_NAME = "textembedding-gecko@003"  # You can specify other models
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Best practice for API keys
GOOGLE_PROJECT = os.environ.get("GOOGLE_PROJECT") # Enter your google project name

def initialize_embeddings():
    """Initializes and returns the Vertex AI embedding model."""
    if not GOOGLE_API_KEY or not GOOGLE_PROJECT:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_PROJECT environment variables must be set.")
    return VertexAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model_name=EMBEDDING_MODEL_NAME,
        project=GOOGLE_PROJECT,
    )


def initialize_vectorstore(embeddings, persist_directory=PERSIST_DIRECTORY):
    """Initializes and returns the Chroma vector store."""
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def embed_documents(directory, embeddings, vector_store):
    """Embeds documents (TXT and PDF) from the given directory."""

    for filename in tqdm(os.listdir(directory), desc="Embedding documents"):
        filepath = os.path.join(directory, filename)
        try:
            if filename.endswith(".txt"):
                with open(filepath, "r") as f:
                    document_content = f.read()
                    vector_store.add_texts(texts=[document_content])

        except Exception as e:  # Handle potential errors during file processing
            logging.error(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    try:
        embeddings = initialize_embeddings()
        vector_store = initialize_vectorstore(embeddings)
        source_path = "chunks"  # Or get this from command-line arguments
        embed_documents(source_path, embeddings, vector_store)
        logging.info("Embedding process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
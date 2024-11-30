
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_vertexai import VertexAIEmbeddings
import os
from langchain.text_splitter import CharacterTextSplitter  # For basic chunking
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants and Configuration
embeddings = VertexAIEmbeddings(google_api_key=constants.google_apikey, model_name="textembedding-gecko@003", project="bot-31-05")
vector_store = Chroma(embedding_function=embeddings, persist_directory="example_data_base")
CHUNK_SIZE = 500  # Adjust as needed
CHUNK_OVERLAP = 50   # Adjust as needed

def hybrid_search(query, top_k, data_directory="example_data"):
    """Performs a hybrid search using semantic and BM25 retrieval."""
    try:
        # Load documents and create chunks (using CharacterTextSplitter for simplicity)
        loader = DirectoryLoader(
            data_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_documents = text_splitter.split_documents(documents)

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(chunked_documents)  # Use chunked docs
        bm25_retriever.k = top_k

        # Create semantic retriever
        vectorstore_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, bm25_retriever], weights=[0.6, 0.4]
        )

        results = ensemble_retriever.invoke(query)
        return results

    except Exception as e:
        logging.error(f"Error during hybrid search: {e}")
        return []  # Return an empty list in case of error


if __name__ == "__main__":
    try:
        query = "Your search query here"  # Replace with user input
        top_k = 3  # Number of results to retrieve

        retrieved_documents = hybrid_search(query, top_k)

        if retrieved_documents:
            for doc in retrieved_documents:
                print(f"Content: {doc.page_content}")
                # Access other metadata like doc.metadata if needed

    except Exception as e:
        logging.error(f"An error occurred: {e}")
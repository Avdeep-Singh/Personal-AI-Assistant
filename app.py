import streamlit as st
import os
import pickle
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

# Load API key from .env file (make sure .env is in .gitignore)
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
os.environ["USER_AGENT"] = "My Personal AI Assistant (your-email@example.com)" 

def vector_embedding():
    """
    Creates and caches vector embeddings for data using pickle for disk caching.
    """
    cache_file = "embeddings_cache.pkl"  # You can customize the cache file name

    if os.path.exists(cache_file):  # Load embeddings from cache if available
        print("--------------------------------------------Embeddings Cached-------------------------------------------------------------")
        with open(cache_file, "rb") as f:
            st.session_state.vectors = pickle.load(f)
    else:   # Calculate and cache embeddings if not available
        print("--------------------------------------------Creating Embeddings-------------------------------------------------------------")
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("<DATA_DIRECTORY>")  # Replace with your data directory 
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20) 
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Save the calculated embeddings to the cache file
        with open(cache_file, "wb") as f:
            pickle.dump(st.session_state.vectors, f)

st.title("AI-Powered Chatbot (Demo)")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.3) 

# Prompt Engineering (Details hidden for privacy)
prompt = ChatPromptTemplate.from_template(
    """
    # Insert your prompt template here. 
    # This section is commented out to protect 
    # sensitive prompt engineering details.
    """
)

prompt1 = st.text_input("Enter your question:")
vector_embedding()

if prompt1 is not None and prompt1.strip() != "":
    if st.button("Ask") or prompt1:
        if len(prompt1) > 60:
            st.warning("Please limit your question to 60 words or less.")
        else:
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                print("Response time :", time.process_time() - start)
                st.write(response['answer'])
            except Exception as e:
                st.error(f"An error occurred: {e}") 
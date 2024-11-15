import os
import gc
import time
import logging
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from operator import itemgetter

import requests
import pickle
import numpy as np
import pandas as pd
import bs4
import faiss  # For creating embeddings
import openai  # Main model
import PyPDF2  # For extracting text from PDFs
from PyPDF2 import PdfReader
from pypdf import PdfReader  # Ensure pypdf is installed
from dotenv import load_dotenv
from tqdm import tqdm

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.utils.math import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain core and community imports
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tracers.log_stream import LogEntry, LogStreamCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ignore warnings
warnings.filterwarnings("ignore")

# from google.cloud import storage
# storage_client = storage.Client()

# bucket_name = "barchart-aichatbot"

# bucket = storage_client.bucket(bucket_name)
# print(f"Bucket {bucket.name} connected.")

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)


print("----------------------------")
print("---Loading Doc_Splits Now---")
print("----------------------------")

# Loading from Doc_Splits

def load_doc_splits(filename='doc_splits.pkl'):
    with open(filename, 'rb') as f:
        doc_splits = pickle.load(f)
    print(f"doc_splits loaded from {filename}")
    return doc_splits

if __name__ == '__main__':
    # Load the doc_splits from the saved file
    doc_splits = load_doc_splits()
    
    print(f"Loaded {len(doc_splits)} chunks from the file.")
    
    
print("------------------------------------")
print("---Creating Vector Embeddings Now---")
print("------------------------------------")


# Creating Vector Embeddings
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def create_faiss_index(doc_splits, batch_size=25, initial_delay=10, max_retries=10, checkpoint_path="Checkpoint/faiss_checkpoint"):
    embeddings = OpenAIEmbeddings()
    vector_db = None
    indexed_docs = 0

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        try:
            vector_db = FAISS.load_local(checkpoint_path, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Loaded checkpoint from {checkpoint_path}.")
            indexed_docs = vector_db.index.ntotal
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}. Starting fresh FAISS indexing.")
    else:
        logging.info("No checkpoint found. Starting fresh FAISS indexing.")

    start_batch = indexed_docs // batch_size

    # Initialize tqdm with the remaining batches to process
    for i in tqdm(range(start_batch * batch_size, len(doc_splits), batch_size), desc="Creating Embeddings", unit="batch", initial=start_batch):
        batch = doc_splits[i:i+batch_size]
        retries = 0

        while retries < max_retries:
            try:
                if vector_db is None:
                    vector_db = FAISS.from_documents(batch, embeddings)
                else:
                    vector_db.add_documents(batch)

                # Save checkpoint after processing each batch
                vector_db.save_local(checkpoint_path)
                logging.info(f"Checkpoint saved at batch {i // batch_size}")

                # Log the number of documents indexed so far
                logging.info(f"Documents indexed so far: {vector_db.index.ntotal}")
                gc.collect()  # Clear unused memory
                break  # Exit retry loop if successful
            except Exception as e:
                if "429" in str(e):
                    wait_time = initial_delay * (2 ** retries)
                    logging.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"An error occurred: {str(e)}")
                    break  # Exit on non-retryable errors

        if retries == max_retries:
            raise Exception("Max retries exceeded. Unable to complete API requests for embeddings.")

    return vector_db

if __name__ == '__main__':

    batch_size = 25
    checkpoint_path = "Checkpoint/faiss_checkpoint"

    try:
        # Create FAISS index with checkpointing
        vector_db = create_faiss_index(doc_splits, batch_size=batch_size, checkpoint_path=checkpoint_path)
        
        # Check the number of indexed documents
        number_of_documents = vector_db.index.ntotal
        logging.info(f"Number of documents in the FAISS index: {number_of_documents}")
        
    except Exception as e:
        logging.error(f"Failed to complete FAISS indexing: {str(e)}")
        

print("------------------------------------------------")
print("---Saving the Vector Embeddings Locally now !--")
print("------------------------------------------------")


vector_path = 'VectorDB'
vectordb_folder = vector_path
index_name="faiss_index"

vector_db.save_local(vectordb_folder, index_name=index_name)
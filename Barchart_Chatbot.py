import openai # main model
import warnings
warnings.filterwarnings("ignore")

import os
import re
import atexit
import base64

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI


# Environment setup
os.environ["OPENAI_API_KEY"] = "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"

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

# Database setup
DATABASE_URL = "sqlite:///chat_history.db"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create the database and the tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

atexit.register(save_all_sessions)

# Specify vector path
vector_path = "/Users/karanmehta/Desktop/University of Chicago/Capstone Project - Barchart/Final Test" # Update this as needed for your directory structure
vectordb_folder = vector_path
index_name = "faiss_index"

# Load the FAISS vector database
embeddings = OpenAIEmbeddings()
new_vector_db = FAISS.load_local(vectordb_folder, embeddings, index_name=index_name, allow_dangerous_deserialization=True)

retriever = new_vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

finance_template = """You are a seasoned finance professional with a keen eye for detail, especially proficient in analyzing SEC filings and calculating key financial ratios of various companies. \
Your expertise allows you to dissect complex financial statements, identify crucial financial metrics, understand the nuances of corporate disclosures, and compute ratios that highlight financial health and efficiency. \
These skills make you an invaluable asset for in-depth financial analysis and advisory roles.
If you don't know the answer, just say that you don't know. \
Answer questions only relevant to the context provided to you. \
Please calculate financial ratios asked in the query and use the relevant data from the context. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", finance_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoke the chain and save the messages after invocation
def invoke_and_save(session_id, input_text):
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)

    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)
    return result


# Load and encode image in base64
file_path = "/Users/karanmehta/Desktop/University of Chicago/Capstone Project - Barchart/Final Test/barchart.png"
with open(file_path, "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode()

# Embed the logo image using HTML with base64 data
st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 10px 0;">
        <img src="data:image/jpeg;base64,{b64_image}" alt="Barchart Logo" width="450">
    </div>
""", unsafe_allow_html=True)

# Set up page layout with custom CSS for matching colors and layout
st.markdown("""
    <style>
    /* Main container styling for Barchart-like layout */
    .main-container {
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
    }

    /* Header */
    .header {
        background-color: #004080; /* Dark blue */
        color: white;
        padding: 10px 0;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }

    /* Chat message styles */
    .user-message, .assistant-message {
        padding: 10px;
        margin: 10px 0;
        font-size: 16px;
        font-family: Arial;
        max-width: 70%;
        overflow-wrap: break-word; /* Ensures long words wrap to the next line */
        word-wrap: break-word;
        word-break: break-word;
        
    }
    
    /* User message styling - aligned right */
    .user-message {
        background-color: #cfe2ff; /* Light blue */
        color: black;
        display: inline-flex;
        border-radius: 18px 18px 4px 18px; /* Rounded corners */
        align-self: flex-end;
        font-family: Arial;
        float: right; /* Align user message to the right without shifting */
        clear: both; /* Ensure each message appears on a new line */
    }

    /* Assistant message styling - aligned left */
    .assistant-message {
        background-color: #e6e6e6; /* Light gray */
        color: black;
        align-self: flex-start;
         border-radius: 18px 18px 18px 4px; /* Rounded corners */
        font-family: Arial;
        float: left; /* Align assistant message to the left */
        clear: both; /* Ensure each message appears on a new line */
    }
    
    /* Container styling to center messages */
    .message-container {
        display: flex;
        flex-direction: column;
        margin: 0 auto;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ffc107; /* Yellow */
        color: #004080; /* Dark blue text */
        border: none;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Chatbox layout styling */
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Chat input */
    .chat-input-container {
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>10K SEC Report - AI Chatbot</div>", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

def format_text_for_consistency(text):
    # Add a space between letters and numbers, but avoid adding spaces with currency symbols and decimal points
    text = re.sub(r'(?<=\D)(?=\d)', r' ', text)  # Add space only if a letter is followed by a number
    text = re.sub(r'(?<=\d)(?=\D)', r' ', text)  # Add space only if a number is followed by a letter
    
    # Remove unwanted spaces between $ sign and numbers, around decimal points, between numbers and commas, and between numbers and question marks
    text = re.sub(r'\$\s*', r'$', text)  # Ensure $ sign is directly followed by numbers
    text = re.sub(r'\s*\.\s*', r'.', text)  # Ensure no spaces around decimal points
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)  # Ensure no spaces between numbers and commas
    text = re.sub(r'(\d)\s*\?\s*', r'\1?', text)  # Ensure no spaces between numbers and question marks
    
    return text

# Main chat container
st.markdown("<div class='chat-container message-container'>", unsafe_allow_html=True)

# Display each message with appropriate styling
for message in st.session_state.messages:
    css_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"<div class='{css_class}'>{format_text_for_consistency(message['content'])}</div>", unsafe_allow_html=True)

# Close the chat container div
st.markdown("</div>", unsafe_allow_html=True)

# Input area for user prompt
if prompt := st.chat_input("Enter your financial question here..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-message'>{format_text_for_consistency(prompt)}</div>", unsafe_allow_html=True)

    # Call the chatbot model API or function (replace invoke_and_save with your actual function)
    response = invoke_and_save(st.session_state.session_id, prompt)

    # Display assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"<div class='assistant-message'>{format_text_for_consistency(response)}</div>", unsafe_allow_html=True)

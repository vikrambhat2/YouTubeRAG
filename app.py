import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from the .env file
load_dotenv()

# Access the variables using os.getenv()
groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")  # Add your Qdrant cluster URL here
qdrant_api_key = os.getenv("Qdrant_API_KEY")

# Initialize the LLM
langchain_llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Global variable to hold the vector database
global_vector_db = None

# Function to process the YouTube URL and create the vector store
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

import logging

logging.basicConfig(level=logging.INFO)

def process_youtube_url(youtube_url):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from {youtube_url}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks")

        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        collection_name = "youtube-video-index"
        qdrant_client.delete_collection(collection_name)
        logging.info(f"Deleted collection '{collection_name}' (if exists)")

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logging.info(f"Created collection '{collection_name}'")

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
        vector_store.add_documents(split_docs)
        logging.info(f"Added {len(split_docs)} documents to collection '{collection_name}'")

        return vector_store
    except Exception as e:
        logging.error(f"Error processing YouTube URL: {str(e)}")
        raise


# Function to answer questions based on the vector store and chat history
def submit_url(youtube_url):
    global global_vector_db
    try:
        # Process the YouTube URL
        global_vector_db = process_youtube_url(youtube_url)
        
        # Default question for summarization
        default_question = "What is the video about"
        chat_history = []  # Empty chat history for the first question
        
        # Get the summary
        summary, _ = answer_question(default_question, chat_history)

        # Status message
        status_message = "Video indexed successfully ✅! You can now ask questions about the video in the chatbot."
    except Exception as e:
        # Handle errors and display in the status box
        status_message = f"❌ Error processing the video: {str(e)}"
        summary = ""  # No summary in case of error
    
    # Return the status message and the summary
    return status_message, summary

def answer_question(question, chat_history):
    global global_vector_db
    if global_vector_db is None:
        return "Please process a YouTube video URL first.", chat_history

    try:
        retriever = global_vector_db.as_retriever(search_kwargs={"k": 5})

        system_prompt = (
            "You are a video assistant tasked with answering questions based on the provided YouTube video context. "
            "Use the given context given by the video author to provide accurate, concise answers in three sentences. "
            "If the context does not contain the answer, say you are not sure "
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(langchain_llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        response = chain.invoke({"input": question})
        
        # Ensure the response contains the expected "answer" key
        if "answer" not in response:
            raise ValueError("Response does not contain an 'answer' key.")
        
        # Append the question and answer to chat history as a tuple
        chat_history.append((question, response['answer']))
        
        # Return the answer and updated chat history
        return response['answer'], chat_history
    except Exception as e:
        # Return error message in case of failure
        error_message = f"Error: {str(e)}"
        chat_history.append((question, error_message))
        return error_message, chat_history


# Gradio interface function to ask a question
def ask_question(question, chat_history):
    response, updated_chat_history = answer_question(question, chat_history)
    return updated_chat_history, updated_chat_history  # Return the updated chat history for gr.Chatbot

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center; color: #4A90E2;'>YouTube Video Q&A</h1>")
        gr.Markdown("<p style='text-align: center;'>Enter a YouTube video URL to extract information and ask questions about it.</p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                youtube_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube video URL here...", lines=1)
                submit_btn = gr.Button("Submit URL", variant="primary")
                
                # Status box for indexing completion or error messages
                status_info = gr.Textbox(
                    label="Status Info", 
                    placeholder="Indexing status will appear here...", 
                    interactive=False, 
                    lines=2
                )

                # Summary box for video summary
                summary_box = gr.Textbox(
                    label="Video Summary", 
                    placeholder="Summary will appear here...", 
                    interactive=False, 
                    lines=6
                )
                
                # Link submit button to the submit_url function
                submit_btn.click(fn=submit_url, inputs=youtube_url, outputs=[status_info, summary_box])

            with gr.Column(scale=1):
                chat_history = gr.Chatbot()
                question = gr.Textbox(label="Your Question", placeholder="Ask a question about the video...", lines=1)
                ask_btn = gr.Button("Ask Question", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                state = gr.State([])

                # Link ask button to the ask_question function
                ask_btn.click(fn=ask_question, inputs=[question, state], outputs=[chat_history, state])
                
                # Clear chat history
                clear_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chat_history, state])  # Clear chat history

        # Add a footer or additional information
        gr.Markdown("<footer style='text-align: center; margin-top: 20px;'>© Vikram Bhat</footer>")

    return demo

# Launch the Gradio interface
interface = create_gradio_interface()
interface.launch()

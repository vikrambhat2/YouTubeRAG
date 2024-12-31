import logging
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_youtube_captions(video_url):
    try:
        # Initialize the YouTube loader
        loader = YoutubeLoader.from_youtube_url(video_url)
        
        # Load the video data and captions
        documents = loader.load()
        
        # Extract and return the captions
        captions = []
        for doc in documents:
            captions.append(f"Document Metadata: {doc.metadata}\nDocument Text: {doc.page_content}\n")
        
        # Calculate the length of the documents
        documents_length = len(documents)
        captions_text = "\n".join(captions)

        if documents_length==0:
            captions_text+="No documents fetched"

        
        return documents_length, captions_text
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None, f"Failed to fetch captions: {str(e)}"

# Create a Streamlit UI
st.title("YouTube Captions Fetcher")
st.write("Enter a YouTube video URL to fetch its captions.")

video_url = st.text_input("YouTube Video URL")

if st.button("Fetch Captions"):
    documents_length, captions_text = fetch_youtube_captions(video_url)
    st.write(f"Number of Documents: {documents_length}")
    st.write("Captions:")
    st.text_area(label="", value=captions_text, height=500)
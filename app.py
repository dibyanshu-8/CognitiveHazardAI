#Import Libraries
import streamlit as st
import pandas as pd
import json
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Page Configuration
st.set_page_config(
    page_title="Cognitive Hazard AI",
    page_icon="🌍",
    layout="wide"
)

#API & model configuration
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except(KeyError,FileNotFoundError):
    st.error("Google API Key not found. Please create a `.streamlit/secrets.toml` file and add your key.")
    st.stop()
    
#We initialize the Gemini Pro model,which will be our AI brain.
llm = genai.GenerativeModel('gemini-1.5-pro-latest')

#Data loading and caching
@st.cache_data
def load_data():
    """Loads supplier and global event data from CSV files."""
    try:
        suppliers_df = pd.read_csv("suppliers.csv")
        events_df = pd.read_csv("global_events.csv")
        return suppliers_df, events_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}.Make sure the CSV files are in the same directory.")
        return None, None
    
#Semantic search engine(vector store) setup
@st.cache_resource
def create_event_retriever(events_df):
    """Creates a FAISS index for fast semantic search of events."""
    model = SentenceTransformer('all-MiniLM-L6-v2') #using a pre-trained model 2 understand meaning of sentences
    event_embeddings = model.encode(events_df['headline'].tolist(), convert_to_tensor=True) #turning all our event headlines into number vectors(embeddings).
    index = faiss.IndexFlatL2(event_embeddings.shape[1]) #creating a FAISS index,which is like a super-fast search engine for these vectors.
    index.add(event_embeddings.cpu().numpy()) #adding our event vectors to this search engine.
    return index, model

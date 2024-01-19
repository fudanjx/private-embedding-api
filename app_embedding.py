import streamlit as st
import requests
import numpy as np
import faiss
import pdfplumber
import os
import pickle

API_URL = "http://dpo.asuscomm.com:8088/predict"
EMBEDDING_FILE = "embeddings.pkl"
CHUNKS_FILE = "chunks.pkl"

@st.cache_data
def convert_pdf_to_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

@st.cache_data
def get_embedding(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, json=payload)
    return response.json()

@st.cache_data
def store_embeddings(embeddings):
    d = len(embeddings[0])  # dimension
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(np.array(embeddings))  # add vectors to the index
    return index

@st.cache_data
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size-overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def query_index(query_text, index, top_k=5):
    query_embedding = get_embedding(query_text)
    D, I = index.search(np.array([query_embedding]), top_k)
    return D, I

st.title('PDF Embedding App')

st.sidebar.header('Upload Document')
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if st.sidebar.button('Upload'):
    if uploaded_file is not None:
        with st.sidebar.container():
            st.write('Converting PDF to text...')
            text = convert_pdf_to_text(uploaded_file)

            st.write('Chunking text...')
            chunks = chunk_text(text)

            st.write('Getting embeddings...')
            embeddings = [get_embedding(chunk) for chunk in chunks]

            st.write('Storing embeddings...')
            index = store_embeddings(embeddings)

            st.write('Saving embeddings and chunks...')
            with open(EMBEDDING_FILE, 'wb') as f:
                pickle.dump(index, f)
            with open(CHUNKS_FILE, 'wb') as f:
                pickle.dump(chunks, f)

            st.success('Done!')

question = st.text_input('Ask a question about the document')

if question:
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(CHUNKS_FILE):
        with st.spinner('Loading embeddings and chunks...'):
            with open(EMBEDDING_FILE, 'rb') as f:
                index = pickle.load(f)
            with open(CHUNKS_FILE, 'rb') as f:
                chunks = pickle.load(f)

        with st.spinner('Searching for answers...'):
            D, I = query_index(question, index)
            st.write('Top 5 chunks:')
            for i in I[0]:
                st.write(chunks[i])
    else:
        st.error('Please upload a document to create the database.')
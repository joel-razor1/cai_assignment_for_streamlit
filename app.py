import streamlit as st
import os
import pandas as pd
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from gpt4all import GPT4All
#from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
import wget
url = 'https://huggingface.co/macoron/ggml-gpt4all-j-v1.3-groovy/resolve/main/ggml-gpt4all-j-v1.3-groovy.bin'
filename = wget.download(url)
# Load the embedding model (open-source)
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load financial statements from PDF
def load_financial_data(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Preprocess & chunk data
def preprocess_data(text, chunk_size=32):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Streamlit UI
st.title("Financial RAG Model")
uploaded_file = st.file_uploader("Upload a financial statement PDF", type=["pdf"])

if uploaded_file is not None:
    financial_text = load_financial_data(uploaded_file)
    text_chunks = preprocess_data(financial_text)
    chunk_embeddings = embedding_model.embed_documents(text_chunks)

    # Store in FAISS (Vector Database)
    vector_store = FAISS.from_texts(text_chunks, embedding_model)

    # BM25 Indexing
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

    # Small open-source LLM (GPT4All - 7B model)
    #llm = GPT4All(model_name="ggml-gpt4all-j-v1.3-groovy.bin", model_path="E:\MultiStage_RAG\\venv\Lib\site-packages\gpt4all")
    llm = GPT4All(model_name=filename, model_path="/mount/src/conai_rag")

    # Input guardrail: Basic validation
    def validate_query(query):
        if any(word in query.lower() for word in ['hack', 'malicious', 'illegal']):
            return "Invalid query. Please ask financial-related questions."
        return query

    # Multi-stage retrieval function
    def retrieve_documents(query, top_k=5):
        bm25_results = bm25.get_top_n(query.split(), text_chunks, n=top_k)
        faiss_results = vector_store.similarity_search(query, k=top_k)
        combined_results = list(set(bm25_results + [doc.page_content for doc in faiss_results]))
        return combined_results[:top_k]

    user_query = st.text_input("Ask a financial question:")

    if user_query:
        validated_query = validate_query(user_query)
        if validated_query.startswith("Invalid"):
            st.write(validated_query)
        else:
            retrieved_docs = retrieve_documents(validated_query)
            context = "\n".join(retrieved_docs)
            prompt_template = PromptTemplate.from_template("""
            Given the financial statements:
            {context}
            Answer the following question: {question}
            """)
            response = llm.generate(prompt_template.format(context=context, question=validated_query),streaming=True, n_ctx = 32)
            st.write(response)

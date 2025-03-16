import streamlit as st
import os
import pandas as pd
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gpt4all import GPT4All

# -------------------------------
# Utility Functions
# -------------------------------

def load_financial_data(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def preprocess_data(text, chunk_size=32):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# -------------------------------
# Load and preprocess the PDF data
# -------------------------------

# Change the path if needed. Here we assume the file is in a folder called "data"
pdf_path = os.path.join("data", "JPMorgan.pdf")
financial_text = load_financial_data(pdf_path)
text_chunks = preprocess_data(financial_text)

# -------------------------------
# Initialize Models and Indices
# -------------------------------

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create vector store and BM25 index
chunk_embeddings = embedding_model.embed_documents(text_chunks)
vector_store = FAISS.from_texts(text_chunks, embedding_model)
bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

# Load GPT4All model
# Ensure your model binary (e.g., "ggml-gpt4all-j-v1.3-groovy.bin") is stored in the "models" folder
llm = GPT4All(model_name="ggml-gpt4all-j-v1.3-groovy.bin", model_path="models")

# -------------------------------
# Query Processing Functions
# -------------------------------

def validate_query(query):
    if any(word in query.lower() for word in ['hack', 'malicious', 'illegal']):
        return "Invalid query. Please ask financial-related questions."
    return query

def retrieve_documents(query, top_k=5):
    bm25_results = bm25.get_top_n(query.split(), text_chunks, n=top_k)
    faiss_results = vector_store.similarity_search(query, k=top_k)
    combined_results = list(set(bm25_results + [doc.page_content for doc in faiss_results]))
    return combined_results[:top_k]

def compute_confidence(query, retrieved_docs):
    query_tokens = query.split()
    bm25_all_scores = bm25.get_scores(query_tokens)
    scores = []
    for doc in retrieved_docs:
        try:
            idx = text_chunks.index(doc)
            scores.append(bm25_all_scores[idx])
        except ValueError:
            continue
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(bm25_all_scores) if bm25_all_scores.size > 0 else 1.0
        confidence = avg_score / max_score  # normalized to [0, 1]
        return confidence
    else:
        return 0.0

prompt_template = PromptTemplate.from_template("""
Given the financial statements:
{context}
Answer the following question: {question}
""")

def generate_response(query):
    validated_query = validate_query(query)
    retrieved_docs = retrieve_documents(validated_query)
    context_text = "\n".join(retrieved_docs)
    prompt = prompt_template.format(context=context_text, question=validated_query)
    response = llm.generate(prompt, streaming=False, n_ctx=32)
    confidence = compute_confidence(validated_query, retrieved_docs)
    return response, confidence

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Financial Statement Query App")
st.write("Ask a question about the financial statements:")

user_query = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_query:
        with st.spinner("Processing..."):
            response, confidence = generate_response(user_query)
        st.subheader("Response:")
        st.write(response)
        st.subheader("Confidence Score:")
        st.write(confidence)
    else:
        st.error("Please enter a query.")

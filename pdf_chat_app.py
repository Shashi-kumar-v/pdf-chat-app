import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import os

access_token = "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"
llm_pipeline = pipeline("text-generation", model="gpt2", token=access_token)

def load_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def search_relevant_text(user_query, document_text):
    start_index = document_text.lower().find(user_query.lower())
    if start_index == -1:
        return "Sorry, no relevant information found."
    end_index = start_index + 500
    return document_text[start_index:end_index]

def get_llm_response(user_query, relevant_text, llm_pipeline):
    prompt = f"{relevant_text}\n\nUser Query: {user_query}\nAnswer:"
    response = llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1)
    return response[0]['generated_text']

st.title("PDF Chat Application")
st.write("Upload a PDF and ask questions about its content.")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    document_text = load_pdf(pdf_file)
    st.write("PDF text loaded successfully!")
    
    user_query = st.text_input("Enter your question:")
    
    if user_query:
        relevant_text = search_relevant_text(user_query, document_text)
        answer = get_llm_response(user_query, relevant_text, llm_pipeline)
        
        st.write("**Relevant Text:**", relevant_text)
        st.write("**Answer:**", answer)

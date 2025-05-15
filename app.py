import os
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# 🧠 Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 📄 Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

# 📚 Chunk text into manageable pieces
def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# 🧠 Build FAISS index
def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# 🔍 Retrieve top-k relevant chunks
def retrieve_top_chunks(query, chunks, index, model, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0]]

# 🧠 Use OpenRouter to generate response
def call_openrouter(messages, model_name="mistralai/mistral-7b-instruct"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model_name,
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()["choices"][0]["message"]["content"]

# 🤖 Intent classification (casual vs question)
def classify_intent(message):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier. Classify the user's message as either "
                    "'question' if it asks about a document, or 'casual' if it's just a polite or informal message."
                )
            },
            {
                "role": "user",
                "content": f"Message: '{message}'\n\nAnswer with only one word: 'question' or 'casual'."
            }
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()["choices"][0]["message"]["content"].strip().lower()

# 💬 Streamlit UI
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Store session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistralai/mistral-7b-instruct"
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = load_model()

# 📤 Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(pdf_file)
        st.session_state.chunks = chunk_text(text)
        st.session_state.index, _ = build_faiss_index(st.session_state.chunks, st.session_state.model)
    st.success("PDF processed. You can now chat below!")

# 📝 Chat input
user_input = st.chat_input("Ask something about your PDF")

# 🗨️ Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🤖 Process new user input
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Classify intent
    with st.spinner("Classifying intent..."):
        intent = classify_intent(user_input)

    # 🧠 Handle casual messages with LLM-generated responses
    if intent == "casual":
        casual_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a friendly assistant. If a message is casual (like greetings, thanks, or small talk), "
                    "respond naturally and politely without referencing the PDF unless the user asks. "
                    "Keep responses short, warm, and engaging."
                )
            },
            {"role": "user", "content": user_input}
        ]
        with st.spinner("Responding casually..."):
            response = call_openrouter(casual_prompt, st.session_state.model_name)

    # 🧠 Handle document questions using RAG
    else:
        top_chunks = retrieve_top_chunks(user_input, st.session_state.chunks, st.session_state.index, st.session_state.model)
        context = "\n\n".join(top_chunks)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]
        with st.spinner("Thinking..."):
            response = call_openrouter(messages, st.session_state.model_name)

    # Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # 📌 Sidebar
with st.sidebar:
    st.markdown("## 🧾 About the App")
    st.info("Upload a PDF and ask questions — this app retrieves relevant chunks using semantic search and answers using OpenRouter LLMs.")

    st.markdown("### 🛠️ How it Works")
    st.markdown("""
    1. 📤 Upload a PDF  
    2. 💬 Ask a question in natural language  
    3. 🔍 Finds relevant parts using FAISS  
    4. 🧠 Answers with an LLM via OpenRouter
    """)

    st.markdown("### 🤖 Powered by")
    st.markdown("- SentenceTransformers (MiniLM-L6-v2)\n- FAISS\n- OpenRouter API\n- Streamlit")

    st.markdown("---")
    st.markdown("👩‍💻 **Built by Ananya**")


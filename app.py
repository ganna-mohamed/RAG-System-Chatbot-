import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import torch
import hashlib
import os
from io import BytesIO

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# Utilities
def file_hash_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

@st.cache_resource
def load_embeddings_and_generator():
    """Load embeddings model and text generator pipeline once (cached)."""
    # embeddings (LangChain wrapper around sentence-transformers)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        embeddings = None

    # generator
    try:
        device = 0 if torch.cuda.is_available() else -1
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device,
            max_length=400,
            truncation=True
        )
    except Exception as e:
        st.error(f"Failed to load text generator pipeline: {e}\nTrying to continue without generator.")
        generator = None

    return embeddings, generator

embeddings, generator = load_embeddings_and_generator()

def process_pdf_bytes(file_bytes: bytes, filename: str, chunk_size=800, chunk_overlap=100):
    """Process PDF from bytes: extract text, split, build FAISS vectorstore and retriever."""
    try:
        pdf_reader = PdfReader(BytesIO(file_bytes))
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        if not text:
            return False, "No readable text found in PDF (scanned image PDF?)."

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        if not embeddings:
            return False, "Embeddings model not loaded."

        try:
            vectorstore = FAISS.from_texts(chunks, embeddings)
        except Exception as e:
            return False, f"Failed building FAISS vectorstore: {e}"

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        # update session state
        st.session_state.retriever = retriever
        st.session_state.generator = generator
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_processed = True
        st.session_state.current_file = filename
        st.session_state.file_hash = file_hash_bytes(file_bytes)
        return True, len(chunks)
    except Exception as e:
        return False, str(e)

def answer_question(question: str, max_length=500):
    if st.session_state.retriever is None:
        return "Please upload a PDF first."
    if st.session_state.generator is None:
        return "Generator model not available."

    try:
        docs = st.session_state.retriever.get_relevant_documents(question)
    except Exception as e:
        return f"Error retrieving documents: {e}"

    if not docs:
        return "I couldn't find relevant information in the PDF to answer this question."

    # build context (limit total length a bit)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""Based on the following context from the document, provide a comprehensive and detailed answer to the question.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

Please provide a thorough answer that:
1. Directly addresses the question using information from the context
2. Includes specific details, examples, and explanations
3. Is well-structured and easy to understand
4. If the context doesn't contain enough information, clearly state what limitations exist

DETAILED ANSWER:"""

    try:
        # generator parameters
        res = st.session_state.generator(
            prompt,
            max_length=max_length,
            do_sample=False,
            temperature=0.3,
            repetition_penalty=1.1
        )
        answer = res[0].get("generated_text") or res[0].get("summary_text") or str(res[0])
        # cleanup
        if "DETAILED ANSWER:" in answer:
            answer = answer.split("DETAILED ANSWER:")[-1].strip()
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

def show_source_documents(question: str):
    if st.session_state.retriever:
        docs = st.session_state.retriever.get_relevant_documents(question)
        with st.expander("🔍 View Source Documents", expanded=False):
            for i, doc in enumerate(docs):
                st.write(f"*Source {i+1}:*")
                text_preview = doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else "")
                st.text(text_preview)
                st.divider()

def clear_chat():
    st.session_state.messages = []

def reset_system():
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.vectorstore = None
    st.session_state.current_file = None
    st.session_state.file_hash = None


# Sidebar
with st.sidebar:
    st.title("📁 PDF Upload")
    st.write("Upload a text-based PDF (not scanned images).")
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

    st.markdown("---")
    st.write("*Controls*")
    if st.button("🗑 Clear Chat", use_container_width=True):
        clear_chat()
        st.experimental_rerun()

    if st.button("🔄 New PDF", use_container_width=True):
        reset_system()
        st.experimental_rerun()

    st.checkbox("Show source documents after answer", value=st.session_state.show_sources, key="show_sources")

    st.markdown("---")
    st.write("⚠ If your PDF is a scanned image PDF, OCR is required (not included here).")
    st.write("Run this app with: streamlit run app.py")


# Main UI
st.title("🤖 PDF Chatbot")
st.write("*Instant processing • Detailed answers • Source tracking*")

# show messages
for message in st.session_state.messages:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(message.get("content", ""))

# handle upload
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    new_hash = file_hash_bytes(file_bytes)
    same_file = (st.session_state.file_hash == new_hash)
    if not st.session_state.pdf_processed or not same_file:
        with st.spinner("⚡ Processing PDF..."):
            success, result = process_pdf_bytes(file_bytes, uploaded_file.name)
        if success:
            st.success("✅ PDF Ready!")
            st.info(f"{result} text chunks** processed from *{uploaded_file.name}*")
            welcome_msg = f"Hello! I've processed *{uploaded_file.name}* with *{result} text chunks*. Ask me anything about the content!"
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        else:
            st.error(f"❌ Processing failed: {result}")

if st.session_state.pdf_processed:
    # input
    prompt = st.chat_input("Ask a question about your PDF... (press Enter to send)")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                response = answer_question(prompt)
            st.markdown(response)
            if st.session_state.show_sources:
                show_source_documents(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()
else:
    # disabled input
    st.chat_input("Upload a PDF to start chatting...", disabled=True)
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("👋 *Welcome to PDF Chatbot!*")
            st.markdown("""
            I can help you analyze and answer questions about your PDF documents.
            *Get started:* Upload a text-based PDF using the sidebar and start asking questions!
            """)

# styling (optional)
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stChatInput {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
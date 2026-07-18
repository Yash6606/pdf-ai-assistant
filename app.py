import os
import io
import fitz
import pytesseract
import streamlit as st

from PIL import Image

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# ======================
# CONFIG
# ======================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-or-v1-xxxx") # change this to your API key or set it in your environment

MODEL_ID = "deepseek/deepseek-chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # change this to your tesseract path


# ======================
# PROMPT
# ======================

template = """
You are a helpful PDF assistant.

Rules:
- Answer ONLY using the context
- If not found say: "Not found in the document"

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# ======================
# OCR
# ======================

def extract_images_with_ocr(pdf_path):

    ocr_data = {}
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):

                images = page.get_images(full=True)

                texts = []

                for img in images:

                    xref = img[0]

                    base_image = doc.extract_image(xref)

                    image = Image.open(io.BytesIO(base_image["image"]))

                    text = pytesseract.image_to_string(image)

                    if text.strip():
                        texts.append(text)

                if texts:
                    ocr_data[i] = "\n".join(texts)
    except Exception as e:
        st.warning(
            f"OCR processing failed or Tesseract is not configured: {e}. "
            "Proceeding with standard text extraction."
        )
        return {}

    return ocr_data


# ======================
# LOAD PDF
# ======================

def load_pdf_text(pdf_path):

    loader = PyPDFLoader(pdf_path)

    return loader.load()


# ======================
# MERGE OCR + TEXT
# ======================

def merge_text_and_ocr(text_docs, ocr_data, pdf_path):

    merged = []

    for doc in text_docs:

        page = doc.metadata.get("page", 0)

        content = doc.page_content

        if page in ocr_data:

            content += "\n\n[OCR TEXT]\n" + ocr_data[page]

        merged.append(
            Document(
                page_content=content,
                metadata={
                    "page": page,
                    "source": pdf_path
                }
            )
        )

    return merged


# ======================
# SEMANTIC CHUNKING
# ======================

def chunk_documents(docs):

    embeddings = load_embeddings()

    splitter = SemanticChunker(embeddings)

    return splitter.split_documents(docs)


# ======================
# VECTORSTORE SAVE / LOAD
# ======================

def load_or_create_vectorstore(chunks, vector_dir):

    embeddings = load_embeddings()

    if os.path.exists(vector_dir):

        vectorstore = FAISS.load_local(
            vector_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        vectorstore.save_local(vector_dir)

    return vectorstore


# ======================
# BM25
# ======================

def create_bm25(chunks):

    texts = [doc.page_content for doc in chunks]

    tokenized = [t.split() for t in texts]

    bm25 = BM25Okapi(tokenized)

    return bm25


# ======================
# HYBRID SEARCH
# ======================

def hybrid_search(query, vectorstore, bm25, docs):

    vector_docs = vectorstore.similarity_search(query, k=5)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:5]

    keyword_docs = [docs[i] for i in top_n]

    combined = vector_docs + keyword_docs

    unique_docs = list({d.page_content: d for d in combined}.values())

    return unique_docs


# ======================
# MODELS
# ======================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    """Load the embedding model once per Streamlit server process."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner="Loading reranking model...")
def load_reranker():
    """Load the optional reranker lazily so the UI can start under low memory."""
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"), None
    except OSError as exc:
        return None, str(exc)


def rerank_docs(query, docs):

    reranker, load_error = load_reranker()

    if reranker is None:
        st.warning(
            "The reranking model could not be loaded because Windows is low on "
            "virtual memory. Showing the search results without reranking."
        )
        return docs[:3]

    pairs = [(query, d.page_content) for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:3]]


# ======================
# LLM
# ======================

def load_llm():

    llm = ChatOpenAI(
        model=MODEL_ID,
        temperature=0.2,
        max_tokens=1024,
        base_url=OPENROUTER_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "PDF Assistant"
        }
    )

    return llm


# ======================
# GENERATE ANSWER
# ======================

def generate_answer(llm, query, docs):

    context = "\n\n".join([d.page_content for d in docs])

    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(formatted_prompt)

    return response.content


# ======================
# STREAMLIT UI
# ======================

st.set_page_config(
    page_title="AI PDF Assistant",
    layout="wide"
)

st.title("📚 AI PDF Assistant")

# Sidebar for API Key configuration
st.sidebar.title("Configuration")
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key or api_key == "sk-or-v1-xxxx":
    user_api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        st.sidebar.success("API Key updated!")
    else:
        st.sidebar.warning("Please enter your API Key to enable the AI assistant.")

# chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_path = f"temp_{uploaded_file.name}"
    vector_dir = f"vectorstore_{uploaded_file.name}"

    # Cache processing to avoid running on every rerun/keystroke
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("Processing PDF (OCR & Vector indexing)..."):
            text_docs = load_pdf_text(pdf_path)
            ocr_data = extract_images_with_ocr(pdf_path)
            merged_docs = merge_text_and_ocr(
                text_docs,
                ocr_data,
                pdf_path
            )
            chunks = chunk_documents(merged_docs)
            vectorstore = load_or_create_vectorstore(
                chunks,
                vector_dir
            )
            bm25 = create_bm25(chunks)

            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.bm25 = bm25
            st.session_state.chunks = chunks
            st.session_state.processed_file = uploaded_file.name

        try:
            os.remove(pdf_path)
        except Exception:
            pass

    st.success("PDF ready!")

    # Verify if API Key is available before showing query input
    current_key = os.environ.get("OPENAI_API_KEY", "")
    if not current_key or current_key == "sk-or-v1-xxxx":
        st.info("Please enter your OpenRouter API Key in the sidebar to ask questions.")
    else:
        query = st.text_input("Ask a question")

        if query:
            vectorstore = st.session_state.vectorstore
            bm25 = st.session_state.bm25
            chunks = st.session_state.chunks

            with st.spinner("Thinking..."):
                llm = load_llm()
                docs = hybrid_search(query, vectorstore, bm25, chunks)
                docs = rerank_docs(query, docs)
                answer = generate_answer(llm, query, docs)

            st.session_state.chat_history.append({
                "question": query,
                "answer": answer
            })

        # Show the current conversation first, followed by older history.
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**AI:** {chat['answer']}")

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

os.environ["OPENAI_API_KEY"] = "sk-or-v1-xxxx" # change this to your API key

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

    doc = fitz.open(pdf_path)
    ocr_data = {}

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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(embeddings)

    return splitter.split_documents(docs)


# ======================
# VECTORSTORE SAVE / LOAD
# ======================

def load_or_create_vectorstore(chunks, vector_dir):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
# RERANKER
# ======================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_docs(query, docs):

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

# chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    vector_dir = f"vectorstore_{uploaded_file.name}"

    with st.spinner("Processing PDF..."):

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

        llm = load_llm()

    st.success("PDF ready!")

    query = st.text_input("Ask a question")

    if query:

        docs = hybrid_search(query, vectorstore, bm25, chunks)

        docs = rerank_docs(query, docs)

        answer = generate_answer(llm, query, docs)

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer
        })

    for chat in st.session_state.chat_history:

        st.markdown(f"**You:** {chat['question']}")

        st.markdown(f"**AI:** {chat['answer']}")

    os.remove(pdf_path)
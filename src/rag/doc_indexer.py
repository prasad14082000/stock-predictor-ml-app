import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def build_vector_index_from_pdfs(uploaded_files, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Takes a list of uploaded file-like objects (from Streamlit), indexes them in-memory, and returns a Chroma vectorstore.
    """
    all_chunks = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location, then process
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp:
            temp.write(uploaded_file.read())
            temp.flush()
            # Use PyPDFLoader to load text
            try:
                loader = PyPDFLoader(temp.name)
                docs = loader.load()
            except Exception as e:
                print(f"Failed to load {uploaded_file.name}: {e}")
                continue

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks found in uploaded documents.")

    # In-memory vector store (DO NOT set persist_directory)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(all_chunks, embedding=embeddings)

    return vectordb


'''
How It Works:
User uploads PDFs. Each file is stored temporarily (never saved to disk for long).
Each is loaded, split, and chunked.
All chunks are indexed in-memory (not persisted).
If you restart the app, the index is rebuilt, so no privacy issues.'''
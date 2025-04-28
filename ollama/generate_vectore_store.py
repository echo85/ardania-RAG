# Langchain dependencies
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)

vector_store = Chroma(
    collection_name="ardania_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Directory to your pdf files:
DATA_PATH = "./data/"


def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.
    Returns:
    List of Document objects: Loaded PDF documents represented as Langchain
                                                            Document objects.
    """
    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    # Load PDF documents and return them as a list of Document objects
    return document_loader.load()


documents = load_documents()  # Call the function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Size of each chunk in characters
    chunk_overlap=200,  # Overlap between consecutive chunks
    length_function=len,  # Function to compute the length of the text
    add_start_index=True,  # Flag to add start index to each chunk
)
all_splits = text_splitter.split_documents(documents)
vector_store.add_documents(all_splits)  # Add documents to the vector store
print("Vetor store created and documents added successfully.")

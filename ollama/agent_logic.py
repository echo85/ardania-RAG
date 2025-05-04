from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_transformers import LongContextReorder
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# Ensure Langsmith tracing is optional or configurable for deployment
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")

# Get config from environment variables
OLAMA_API_KEY = os.environ.get(
    "OLAMA_API_KEY"
)  # May not be needed depending on Ollama setup
OLAMA_API_URL = os.environ.get(
    "OLAMA_API_URL", "http://localhost:11434"
)  # Default to local if not set
MODEL = os.environ.get("MODEL", "gemma3:12b")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "jeffh/intfloat-multilingual-e5-large:f16"
)
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db_ardania")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.76))
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", 25))

PROMPT_TEMPLATE = """
Sei un assistente che fornisce informazioni sul mondo di Ardania GDR Ultima On Line.
Rispondi alla domanda in base al contesto fornito di seguito:
{context}
- -
Rispondi alla domanda in base al contesto fornito sopra: {question} /no_think
"""


# --- State Definition ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# --- Core Logic Functions ---
def initialize_components():
    """Initializes and returns the core components (LLM, Embeddings, Vector Store)."""
    headers = {}
    if OLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLAMA_API_KEY}"

    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLAMA_API_URL,
            client_kwargs={"headers": headers},  # Pass headers if needed
        )
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama Embeddings at {OLAMA_API_URL} for model {EMBEDDING_MODEL}: {e}"
        )

    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
    except Exception as e:
        # Provide more specific error handling if possible (e.g., check if path exists)
        raise ConnectionError(f"Failed to load ChromaDB from {CHROMA_DB_PATH}: {e}")

    try:
        llm = ChatOllama(
            base_url=OLAMA_API_URL,
            client_kwargs={"headers": headers},  # Pass headers if needed
            model=MODEL,
            temperature=0,
        )
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama LLM at {OLAMA_API_URL} for model {MODEL}: {e}"
        )

    return llm, embeddings, vector_store


def retrieve(state: State, embeddings: OllamaEmbeddings, vector_store: Chroma) -> State:
    """Retrieves relevant documents from the vector store."""
    print(f"--- Retrieving documents for question: {state['question']} ---")
    _filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=SIMILARITY_THRESHOLD
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_filter,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K}),
    )
    compressed_docs = compression_retriever.invoke(state["question"])
    # compressed_docs = vector_store.similarity_search(state["question"], k=RETRIEVER_K)
    print(f"--- Retrieved {len(compressed_docs)} documents ---")
    state["context"] = compressed_docs
    return state


def generate(state: State, llm: ChatOllama) -> State:
    """Generates an answer based on the retrieved context and question."""
    print(f"--- Generating answer for question: {state['question']} ---")
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(state["context"])

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = create_stuff_documents_chain(llm, prompt_template)
    response = chain.invoke({"context": reordered_docs, "question": state["question"]})
    print(f"--- Generated answer: {response[:100]}... ---")  # Log snippet
    state["answer"] = response
    return state


def build_graph(llm: ChatOllama, embeddings: OllamaEmbeddings, vector_store: Chroma):
    """Builds the LangGraph."""
    graph_builder = StateGraph(State)

    # Bind components to the node functions
    graph_builder.add_node(
        "retrieve", lambda state: retrieve(state, embeddings, vector_store)
    )
    graph_builder.add_node("generate", lambda state: generate(state, llm))

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "__end__")  # Use standard end node

    graph = graph_builder.compile()
    print("--- Graph compiled successfully ---")
    return graph


# --- Example Usage (Optional, for testing) ---
if __name__ == "__main__":
    try:
        print("Initializing components...")
        llm_instance, embeddings_instance, vector_store_instance = (
            initialize_components()
        )
        print("Building graph...")
        graph_instance = build_graph(
            llm_instance, embeddings_instance, vector_store_instance
        )
        print("Graph ready.")

        # Test invocation
        test_question = "Dove è Hammerheim?"
        print(f"\nInvoking graph with question: {test_question}")
        final_state = graph_instance.invoke({"question": test_question})
        print(f"\nFinal Answer: {final_state['answer']}")

        # Test streaming
        print(f"\nStreaming graph with question: {test_question}")
        for chunk in graph_instance.stream({"question": test_question}):
            print(chunk)

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

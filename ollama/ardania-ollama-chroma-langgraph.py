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

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")

OLAMA_API_KEY = os.environ.get("OLAMA_API_KEY")
OLAMA_API_URL = os.environ.get("OLAMA_API_URL")
MODEL = "gemma3:12b"
EMBEDDING_MODEL = "jeffh/intfloat-multilingual-e5-large:f16"

headers = {
    "Authorization": f"Bearer {OLAMA_API_KEY}",
}

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLAMA_API_URL,
    client_kwargs={"headers": headers},
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_ardania",  # Where to save data locally, remove if not necessary
)

llm = ChatOllama(
    base_url=OLAMA_API_URL,
    client_kwargs={"headers": headers},
    model=MODEL,
    temperature=0,
)


PROMPT_TEMPLATE = """
Sei un assistente che fornisce informazioni sul mondo di Ardania GDR Ultima On Line.
Rispondi alla domanda in base al contesto fornito di seguito:
{context}
- -
Rispondi alla domanda in base al contesto fornito sopra: {question} /no_think
"""


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(State: State) -> State:
    _filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_filter,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 30}),
    )
    compressed_docs = compression_retriever.invoke(State["question"])

    # results = vector_store.similarity_search_with_score(State["question"], k=50)
    State["context"] = compressed_docs
    return State


def generate(State: State) -> State:
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(State["context"])
    """ sources_formatted = "=================\n".join(
        [res.page_content for res in reordered_docs]
    ) """

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = create_stuff_documents_chain(llm, prompt_template)
    response = chain.invoke({"context": reordered_docs, "question": State["question"]})
    """ message = prompt_template.invoke(
        {"question": State["question"], "context": sources_formatted}
    ) """

    # response = llm.invoke(message)
    State["answer"] = response
    return State


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def main():
    try:
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            for message, metadata in graph.stream(
                {"question": input_text}, stream_mode="messages"
            ):
                if metadata["langgraph_node"] == "generate":
                    print(message.content, end="")
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()

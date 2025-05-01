from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

embeddings = OllamaEmbeddings(
    model="DC1LEX/nomic-embed-text-v1.5-multimodal:latest",
)

vector_store = Chroma(
    collection_name="ardania_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_dbnomic15",  # Where to save data locally, remove if not necessary
)

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

question = "Quale è la citta più vicina ad Hammerheim?"
PROMPT_TEMPLATE = """
Sei un assistente che fornisce informazioni sul mondo di Ardania GDR Ultima On Line.
Rispondi alla domanda in base al contesto fornito di seguito:

{context}
- -
Rispondi alla domanda in base al contesto fornito sopra: {question}
"""

results = vector_store.similarity_search_with_score(question, k=30)
sources_formatted = "=================\n".join(
    [res.page_content for res, _score in results]
)

prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
message = prompt_template.invoke({"question": question, "context": sources_formatted})
print(message)

print("=========================================")
ai_msg = llm.invoke(message)

print(ai_msg.content)

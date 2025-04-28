from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

messages = [
    (
        "system",
        """You are an assistant that provides information about Ardania GDR Ultima On Line World. 
        Answer the query using only the sources provided below.""",
    ),
    ("human", "Dove è Hammerheim?"),
]
ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)

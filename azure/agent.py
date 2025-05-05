# Install the following dependencies: azure.identity and azure-ai-inference
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.ai.inference.models import AssistantMessage

load_dotenv()
endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
model_name = os.getenv("DEPLOYMENT_NAME")
key = os.getenv("AZURE_INFERENCE_SDK_KEY")


# SEARCH SERVICE
index_name = "ardaniamd-index"
index_vector_field = "text_vector"
index_field_content = "chunk"
DEPLOYMENT_NAME = "Phi-4"
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_SERVICE_QUERY_KEY")

azure_credential = AzureKeyCredential(search_key)
search_client = SearchClient(
    endpoint=search_endpoint, index_name=index_name, credential=azure_credential
)

def retrieve(user_input: str) -> str:
    vector_query = VectorizableTextQuery(
        text=user_input, k_nearest_neighbors=30, fields=index_vector_field
    )
    search_results = search_client.search(
        search_text=user_input,
        vector_queries=[vector_query],
        select=[index_field_content],
        top=5,
    )

    sources_formatted = "=================\n".join(
        [
            f"{document[index_field_content]}"
            for document in search_results
        ]
    )

    return sources_formatted


def get_response(user_input: str) -> str:
    user_msg = UserMessage(content=user_input)
    system_msg = SystemMessage(
        content="""Sei un assistente che fornisce informazioni sul mondo di Ardania GDR Ultima On Line.
        Rispondi alla domanda in base al contesto fornito di seguito:"""
    )

    chat_messages = [system_msg, user_msg]
    sources_formatted = retrieve(user_input)
    chat_messages.append(SystemMessage(content=f"Sources:\n {sources_formatted}"))
    client = ChatCompletionsClient(
                endpoint=endpoint, credential=AzureKeyCredential(key)
            )
    result = client.complete(
        model=model_name,
        messages=chat_messages,
        temperature=0.7
    )
    return result.choices[0].message.content

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

            response = get_response(input_text)
            print(response)

            # Add the response to the chat history
            # prompt.append(AssistantMessage(content=response["choices"][0]["message"]["content"]))

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()

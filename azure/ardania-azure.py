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
index_name = "ardania-index-ai"
index_vector_field = "text_vector"
index_field_content = "chunk"
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_SERVICE_QUERY_KEY")

azure_credential = AzureKeyCredential(search_key)
search_client = SearchClient(
    endpoint=search_endpoint, index_name=index_name, credential=azure_credential
)

prompt = [
    SystemMessage(
        content="""You are an assistant that provides information about Ardania GDR Ultima On Line World. 
        Answer the query using only the sources provided below."""
    )
]


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

            prompt.append(
                UserMessage(content=input_text),
            )

            vector_query = VectorizableTextQuery(
                text=input_text, k_nearest_neighbors=50, fields=index_vector_field
            )
            search_results = search_client.search(
                search_text=input_text,
                vector_queries=[vector_query],
                select=[index_field_content],
                top=5,
            )

            sources_formatted = "=================\n".join(
                [
                    f"CONTENT: {document[index_field_content]}"
                    for document in search_results
                ]
            )

            prompt.append(SystemMessage(content=f"Sources:\n {sources_formatted}"))

            client = ChatCompletionsClient(
                endpoint=endpoint, credential=AzureKeyCredential(key)
            )

            response = client.complete(
                messages=prompt,
                model=model_name,
                max_tokens=1000,
            )
            print(response["choices"][0]["message"]["content"])

            # Add the response to the chat history
            prompt.append(
                AssistantMessage(content=response["choices"][0]["message"]["content"])
            )

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()

# Install the following dependencies: azure.identity and azure-ai-inference
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

load_dotenv()

# SEARCH SERVICE
index_name = "ardaniamd-index"
index_vector_field = "text_vector"
index_field_content = "chunk"
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_SERVICE_QUERY_KEY")

azure_credential = AzureKeyCredential(search_key)
search_client = SearchClient(
    endpoint=search_endpoint, index_name=index_name, credential=azure_credential
)


def main():
    try:
        while True:
            # Get input text
            input_text = input("Enter the text to search (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            vector_query = VectorizableTextQuery(
                text=input_text, k_nearest_neighbors=30, fields=index_vector_field
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

            print(sources_formatted)

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()

import os
from ...dotenv import load_dotenv
import requests
import csv
from io import StringIO
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.core import QueryEngine

# Set up OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def fetch_crunchbase_data():
    url = "https://gist.githubusercontent.com/jmperez/5791045/raw/295862e4b91a93157f84b050bb169940b7f4042d/crunchbase-companies.csv"
    response = requests.get(url)
    return response.text

def create_rag_system():
    # Fetch and process the Crunchbase data
    csv_data = fetch_crunchbase_data()
    csv_file = StringIO(csv_data)
    csv_reader = csv.DictReader(csv_file)

    documents = []
    for row in csv_reader:
        content = f"Company: {row['name']}\nCategory: {row['category']}\nDescription: {row['description']}"
        documents.append(Document(text=content))

    # Parse nodes
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)

    # Create index
    index = GPTVectorStoreIndex(nodes)

    return index

# Example usage
print("Creating RAG system from Crunchbase data...")
index = create_rag_system()

# Query the index
query_engine = index.as_query_engine()
query = "What are some popular AI startups?"
try:
    response = query_engine.query(query)
    print("\nRAG Query Result:")
    print(response)
except Exception as e:
    print(f"\nError querying the index: {e}")

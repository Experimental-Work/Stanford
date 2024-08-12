import os
import requests
import csv
from io import StringIO
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document as LangchainDocument
import textwrap

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Set up OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Part 1: Summarization using LangChain
def summarize_text(text, method="stuff"):
    llm = OpenAI(temperature=0, api_key=openai_api_key)
    docs = [LangchainDocument(page_content=text)]

    if method == "stuff":
        chain = load_summarize_chain(llm, chain_type="stuff")
    elif method == "map_reduce":
        chain = load_summarize_chain(llm, chain_type="map_reduce")
    else:
        raise ValueError("Invalid method. Choose 'stuff' or 'map_reduce'.")

    summary = chain.run(docs)
    return summary

# Example usage
text = """
Your long text here. This should be a substantial amount of text that requires summarization.
"""

print("Stuff method summary:")
print(textwrap.fill(summarize_text(text, "stuff"), width=80))

print("\nMap-reduce method summary:")
print(textwrap.fill(summarize_text(text, "map_reduce"), width=80))

# Part 2: Simple RAG system using Crunchbase Open Data Map

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
    parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    )
    nodes = parser.get_nodes_from_documents(documents)

    # Create index
    index = VectorStoreIndex(nodes)

    return index

# Example usage
print("Creating RAG system from Crunchbase data...")
index = create_rag_system()

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are some popular AI startups?")
print("\nRAG Query Result:")
print(textwrap.fill(str(response), width=80))

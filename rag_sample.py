import os
import requests
import csv
from io import StringIO
import textwrap
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

# Set up OpenAI API key
from dotenv import load_dotenv
load_dotenv()

# Part 1: Summarization using llama-index
def summarize_text(text, method="stuff"):
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    
    if method == "stuff":
        response = llm.complete(f"Summarize the following text:\n\n{text}")
    elif method == "map_reduce":
        # For map_reduce, we'll split the text and summarize each part, then combine
        splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = splitter.split_text(text)
        summaries = [llm.complete(f"Summarize the following text:\n\n{chunk}") for chunk in chunks]
        response = llm.complete(f"Combine these summaries into a coherent summary:\n\n{''.join(summaries)}")
    else:
        raise ValueError("Invalid method. Choose 'stuff' or 'map_reduce'.")

    return str(response)

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
    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
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

# Example of using chat with a list of messages
messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant."),
    ChatMessage(role="user", content="Tell me about Paul Graham."),
]

llm = OpenAI()
chat_response = llm.chat(messages)
print("\nChat Response about Paul Graham:")
print(textwrap.fill(str(chat_response), width=80))

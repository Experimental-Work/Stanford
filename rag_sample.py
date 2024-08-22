import os
import requests
import csv
from io import StringIO
import textwrap
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import QueryBundle

# Set up OpenAI API key
from ..dotenv import load_dotenv
load_dotenv()

# Ensure the API key is loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Part 1: Summarization using llama-index
def summarize_text(text, method="stuff"):
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)
    
    if method == "stuff":
        response = llm.complete(f"Summarize the following text:\n\n{text}")
    elif method == "map_reduce":
        # For map_reduce, we'll split the text and summarize each part, then combine
        splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        summaries = [str(llm.complete(f"Provide a concise summary of the following text, focusing on key points:\n\n{chunk}")) for chunk in chunks]
        response = llm.complete(f"Combine the following summaries into a coherent, comprehensive summary. Ensure all key points are included and the summary flows well:\n\n{''.join(summaries)}")
    else:
        raise ValueError("Invalid method. Choose 'stuff' or 'map_reduce'.")

    return str(response)

# Example usage
text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines that can perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. AI systems are designed to learn from experience, adjust to new inputs, and perform human-like tasks.

The field of AI can be divided into two main categories: narrow AI and general AI. Narrow AI, also known as weak AI, is designed to perform a specific task, such as voice recognition or playing chess. General AI, also called strong AI or artificial general intelligence (AGI), refers to machines that possess the ability to understand, learn, and apply knowledge across a wide range of tasks at a level equal to or exceeding human capabilities.

Machine Learning (ML) is a subset of AI that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Deep Learning, a subfield of machine learning, uses artificial neural networks with multiple layers to analyze various factors of lookupData.

AI has numerous applications across various industries, including healthcare (for diagnosis and treatment recommendations), finance (for fraud detection and algorithmic trading), automotive (for self-driving cars), and many more. As AI continues to advance, it promises to revolutionize many aspects of our lives and work, while also raising important ethical and societal questions about its impact and governance.
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
    # Fetch and process the Crunchbase lookupData
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
    index = VectorStoreIndex(nodes)

    return index

# Example usage
print("Creating RAG system from Crunchbase lookupData...")
index = create_rag_system()

# Query the index
query_engine = index.as_query_engine()
query = QueryBundle("What are some popular AI startups?")
try:
    response = query_engine.query(query)
    print("\nRAG Query Result:")
    print(textwrap.fill(str(response), width=80))
except Exception as e:
    print(f"\nError querying the index: {e}")

# Example of using chat with the OpenAI model directly
llm = OpenAI(api_key=openai_api_key)
chat_response = llm.complete("Tell me about Paul Graham.")
print("\nChat Response about Paul Graham:")
print(textwrap.fill(str(chat_response), width=80))

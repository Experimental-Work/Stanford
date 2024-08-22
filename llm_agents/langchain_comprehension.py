import asyncio
import os
import ssl
from io import BytesIO

import aiohttp
import certifi
import tiktoken
from PyPDF2 import PdfReader
from aiohttp import ClientTimeout
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv

# Load encoding
tiktoken.get_encoding("o200k_base")

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)

# Set up OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError(f"No OpenAI API key found. Make sure it's set in your .env file at {dotenv_path}")

llm = ChatOpenAI(api_key=openai_api_key)
llm.temperature = 0
llm.model_name = "gpt-4o"

pdf_urls = [
    "https://www.stepstonegroup.com/wp-content/uploads/2022/11/Venture-Capital_-Partying-Like-Its-1999_.pdf",
    "lookup_data/Ch9Leleux.pdf",  # Local file
    "https://www.cmc.edu/sites/default/files/Bias%20in%20the%20Reporting%20of%20Venture%20Capital%20Performance-%20The%20Disciplinary%20Role%20of%20FOIA%20%20.pdf",
    "https://www.albion.vc/app/uploads/2022/07/SaaS-AlbionVC.pdf",
    "https://saascan.ca/wp-content/uploads/2022/06/The-Metrics-SaaS-Investors-Care-About-Most.pdf",
    "https://nvca.org/wp-content/uploads/2023/10/Q3_2023_PitchBook-NVCA_Venture_Monitor.pdf"
]

# Create a custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_pdf(session, url):
    try:
        timeout = ClientTimeout(total=60)  # 60 seconds timeout
        async with session.get(url, ssl=ssl_context, timeout=timeout) as response:
            if response.status == 200:
                return await response.read()
            else:
                print(f"Failed to fetch {url}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        raise  # Re-raise the exception to trigger a retry


async def load_pdf(url_or_path):
    if url_or_path.startswith('http'):
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            content = await fetch_pdf(session, url_or_path)
            if content:
                try:
                    pdf = PdfReader(BytesIO(content))
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    return [Document(page_content=text, metadata={"source": url_or_path})]
                except Exception as e:
                    print(f"Error processing PDF from {url_or_path}: {str(e)}")
                    return None
            else:
                return None
    else:
        try:
            loader = PyPDFLoader(url_or_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading local PDF {url_or_path}: {str(e)}")
            return None


def summarize_document(docs):
    if not docs:
        return "Unable to summarize document.", "Unable to summarize document."

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Summarize using the "stuff" method
    stuff_chain = load_summarize_chain(llm, chain_type="stuff")
    stuff_summary = stuff_chain.invoke({"input_documents": splits})

    # Summarize using the "map_reduce" method
    map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
    map_reduce_summary = map_reduce_chain.invoke({"input_documents": splits})

    return stuff_summary['output_text'], map_reduce_summary['output_text']


def compare_summaries(stuff_summary, map_reduce_summary):
    comparison_prompt = f"""
    Compare the following two summaries:

    Stuff method summary:
    {stuff_summary}

    Map-reduce method summary:
    {map_reduce_summary}

    Analyze the differences in content, detail, and overall effectiveness for understanding the document.
    """

    message = HumanMessage(content=comparison_prompt)
    response = llm.invoke([message])
    return response.content


async def process_pdf(url_or_path):
    print(f"\nProcessing: {url_or_path}")
    try:
        docs = await load_pdf(url_or_path)
        if docs:
            stuff_summary, map_reduce_summary = summarize_document(docs)

            print("\nSummary using 'stuff' method:")
            print(stuff_summary)

            print("\nSummary using 'map-reduce' method:")
            print(map_reduce_summary)

            comparison = compare_summaries(stuff_summary, map_reduce_summary)

            print("\nComparison of methods:")
            print(comparison)
        else:
            print(f"Failed to load document: {url_or_path}")
    except Exception as e:
        print(f"Error processing {url_or_path}: {str(e)}")


async def main():
    tasks = [process_pdf(url_or_path) for url_or_path in pdf_urls]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

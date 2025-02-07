from flask import Flask, request, jsonify, session, render_template
from threading import Thread
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI  # Import the LLM wrapper
from web_crawler import fetch_all_links_content, fetch_single_url_content, monitor_website  # Import necessary functions
from text_to_doc import text_to_documents
from prompt import get_prompt_template  # Importing the prompt template function
import os
import shutil
from dotenv import load_dotenv
import logging
from flask_cors import CORS
import secrets
from urllib.parse import urlparse, urlunparse
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Ensure it's defined in your environment or .env file.")

# Setup Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

PERSIST_DIRECTORY = "data/chroma"
MONITOR_INTERVAL = 300  # Default monitoring interval in seconds
monitored_sites = {}  # Dictionary to keep track of monitored sites


def get_chroma_client():
    """
    Initialize and return the Chroma vector store client.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(collection_name="website_data", persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

def normalize_url(url):
    parsed_url = urlparse(url)
    return urlunparse(parsed_url._replace(path=parsed_url.path.rstrip('/')))

def is_url_already_stored(url):
    """
    Check if a URL's data is already stored in the ChromaDB.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL's data is already stored, False otherwise.
    """
    normalized_url = normalize_url(url)
    collection = get_chroma_client()
    results = collection.get(
        where={"url": normalized_url},
        limit=1,
        include=["metadatas"]
    )

    if results["ids"]:
        for meta in results["metadatas"]:
            if meta.get("url") == normalized_url:
                return True

            pdf_links = meta.get("pdf_links", [])
            if isinstance(pdf_links, list) and normalized_url in pdf_links:
                return True

            linked_content_str = meta.get("linked_content", "[]")
            linked_content_list = json.loads(linked_content_str)
            for linked in linked_content_list:
                if linked.get("url") == normalized_url:
                    return True

    return False

def update_vector_store(url):
    """
    Fetch data from a website, count the number of links fetched and skipped, 
    update the vector store, and return the count and links.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        dict: Result of the operation with a status message, count, fetched links, and skipped links.
    """
    if is_url_already_stored(url):
        print(f"The URL '{url}' is already stored in the vector database.")
        return {
            "message": f"The URL '{url}' is already stored in the vector database.",
            "count": 0,
            "fetched_links": [],
            "skipped_links": []
        }

    # Fetch all links and content from the given URL
    fetch_result = fetch_all_links_content(url)
    all_content = fetch_result["content"]
    total_links = fetch_result["count"]
    fetched_count = len(fetch_result["fetched_links"])
    skipped_count = len(fetch_result["skipped_links"])
    fetched_links = fetch_result["fetched_links"]
    skipped_links = fetch_result["skipped_links"]

    if not all_content:
        print(f"No content fetched for URL: {url}")
        return {
            "message": f"No content fetched from URL: {url}",
            "fetched_count": fetched_count,
            "skipped_count": skipped_count,
            "fetched_links": fetched_links,
            "skipped_links": skipped_links,
            "total_links"  : total_links 
        }

    vector_store = get_chroma_client()

    # Process fetched content and add to vector store
    for item in all_content:
        text = item.get("content", "")
        metadata = item.get("metadata", {})
        if not text or not metadata:
            print(f"Invalid content or metadata for URL: {url}")
            continue  # Skip processing if content or metadata is invalid

        # Add the main URL to metadata for future reference
        metadata["source_url"] = url

        # Convert text and metadata to documents
        documents = text_to_documents(text, metadata)
        if not documents:
            print(f"No documents prepared for URL: {url}")
            continue

        # Add documents to the vector store in batches
        max_batch_size = 100
        for i in range(0, len(documents), max_batch_size):
            batch = documents[i:i + max_batch_size]
            print(f"Adding batch of {len(batch)} documents for URL: {url}")
            try:
                vector_store.add_documents(batch)
            except Exception as e:
                print(f"Error adding documents for URL: {url}. Exception: {str(e)}")

    # Persist the vector store
    vector_store.persist()

    print(f"Data from the URL '{url}' has been stored.")
    return {
        "message": f"Data from the URL '{url}' has been stored successfully.",
        "fetched_count": fetched_count,
        "skipped_count": skipped_count,
        "fetched_links": fetched_links,
        "skipped_links": skipped_links,
        "total_links"  : total_links
    }


def update_vector_store_one_url(url):
    """
    Fetch data for a single URL and store it in the vector store.

    Args:
        url (str): The URL to process.

    Returns:
        dict: Result of the operation with a status message.
    """
    try:
        # Check if the URL is already stored
        if is_url_already_stored(url):
            return {
                "url": url,
                "message": f"The URL '{url}' is already stored in the vector database."
            }

        # Fetch the content of the given URL using fetch_single_url_content
        fetch_result = fetch_single_url_content(url)

        # Handle fetch errors
        if "error" in fetch_result:
            return {
                "url": url,
                "error": fetch_result["error"],
                "details": fetch_result.get("details", "")
            }

        # Extract content and metadata
        content = fetch_result["content"]
        metadata = fetch_result["metadata"]

        # Prepare vector store and store content
        vector_store = get_chroma_client()
        documents = text_to_documents(content, metadata)

        if documents:
            vector_store.add_documents(documents)
            vector_store.persist()

        return {
            "url": url,
            "message": f"Data from the URL '{url}' has been stored successfully.",
            "content_length": len(content)  # Use actual content length
        }

    except Exception as e:
        # Handle any unexpected errors
        return {
            "url": url,
            "error": f"Failed to process the URL '{url}'",
            "details": str(e)
        }


def start_monitoring(url, interval):
    """
    Start monitoring a website for changes.

    Args:
        url (str): The URL to monitor.
        interval (int): Time in seconds between checks.
    """
    def monitor_task():
        previous_content = fetch_all_links_content(url)
        while monitored_sites.get(url, False):
            time.sleep(interval)
            current_content = fetch_all_links_content(url)
            if current_content != previous_content:
                print(f"Changes detected for {url}. Updating vector store.")
                # Store only the new/changed content
                for text, metadata in current_content:
                    if text not in [doc[0] for doc in previous_content]:  # Check if content is new
                        metadata["url"] = url  # Add URL to metadata
                        documents = text_to_documents(text, metadata)
                        vector_store = get_chroma_client()
                        vector_store.add_documents(documents)
                        vector_store.persist()
                previous_content = current_content
            else:
                print(f"No changes detected for {url}.")
    Thread(target=monitor_task, daemon=True).start()


@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    """
    Flask route to handle chatbot queries and generate a complete response with token statistics.
    """

    try:
        # Parse input JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        query = data.get("query")
        metadata = data.get("metadata", {})
        if not query or not isinstance(query, str):
            return jsonify({"error": "Query must be a non-empty string"}), 400

        # Retrieve the prompt template and configurations
        prompt_data = get_prompt_template(max_tokens=1000, temperature=0.7, num_documents=15, model="gpt-4o-mini")
        prompt_template = prompt_data["prompt_template"]
        configurations = prompt_data["configurations"]


        vector_store = get_chroma_client()
        retriever = vector_store.as_retriever(search_type="mmr")
        retriever.search_kwargs["k"] = configurations["num_documents"]
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return jsonify({
                "query": query,
                "message": "No relevant data is present in the database for this query."
            }), 404

        docs = [doc for doc in docs if hasattr(doc, "page_content")]
        context = "\n".join([doc.page_content for doc in docs])

        # Prepare the prompt text
        organization_name = metadata.get("organization_name", "your organization")
        prompt_text = prompt_template.format(
            organization_name=organization_name,
            context=context,
            query=query
        )

        # Generate response using ChatOpenAI
        llm = ChatOpenAI(
            temperature=configurations["temperature"],
            max_tokens=configurations["max_tokens"],
            model=configurations["model"]
        )
        llm_response = llm.invoke(prompt_text)
        response_text = llm_response.content

        # Calculate token details
        # token_details = count_tokens(
        #     prompt=prompt_text,
        #     response_text=response_text,
        #     max_tokens=configurations["max_tokens"],
        #     model=configurations["model"]
        # )

        # Prepare the response
        response = {
            "query": query,
            # **token_details,
            "response": response_text
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"An unknown error occurred: {str(e)}"}), 500



@app.route("/update-vector-store", methods=["POST"])
def update_vector_store_route():
    """
    Flask route to crawl a website and update the vector store with new data.
    """
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    result = update_vector_store(url)
    return jsonify(result)


@app.route("/monitor-website", methods=["POST"])
def monitor_website_route():
    """
    Flask route to start monitoring a website for changes.
    """
    data = request.get_json()
    url = data.get("url")
    interval = data.get("interval", MONITOR_INTERVAL)

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if url in monitored_sites and monitored_sites[url]:
        return jsonify({"message": f"{url} is already being monitored."})

    monitored_sites[url] = True
    start_monitoring(url, interval)
    return jsonify({"message": f"Started monitoring {url} every {interval} seconds."})


@app.route("/stop-monitoring", methods=["POST"])
def stop_monitoring_route():
    """
    Flask route to stop monitoring a website.
    """
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if url in monitored_sites and monitored_sites[url]:
        monitored_sites[url] = False
        return jsonify({"message": f"Stopped monitoring {url}."})

    return jsonify({"error": f"{url} is not being monitored."}), 400


@app.route("/reset-chroma", methods=["POST"])
def reset_chroma_route():
    """
    Clear the ChromaDB data and reinitialize it.
    """
    try:
        shutil.rmtree("data/chroma", ignore_errors=True)
        print("ChromaDB directory cleared.")
    except Exception as e:
        return jsonify({"error": f"Failed to reset ChromaDB: {str(e)}"}), 500

    vector_store = Chroma(collection_name="website_data", persist_directory="data/chroma")
    return jsonify({"message": "ChromaDB has been reset successfully."})

@app.route("/update-multiple-urls", methods=["POST"])
def update_multiple_urls():
    """
    Flask route to crawl multiple websites and update the vector store with new data for each URL.
    """
    data = request.get_json()
    urls = data.get("urls")

    if not urls or not isinstance(urls, list):
        return jsonify({"error": "A list of URLs is required"}), 400

    response_data = []
    for url in urls:
        try:
            # Fetch and store data for the URL
            result = update_vector_store_one_url(url)
            response_data.append({
                "url": url,
                "update_result": result
            })
        except Exception as e:
            response_data.append({
                "url": url,
                "error": str(e)
            })

    return jsonify(response_data), 200



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)

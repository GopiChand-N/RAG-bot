from langchain.prompts import PromptTemplate
from web_crawler import fetch_all_links_content
import tiktoken

def get_prompt_template(max_tokens=100, temperature=0.9, num_documents=2, model = "gpt-3.5-turbo"):
    """
    Create and return a prompt template for the chatbot with specific configurations.

    Args:
        max_tokens (int): The maximum number of tokens for the response.
        temperature (float): The creativity level of the response.
        num_documents (int): The number of documents to retrieve for the query.

    Returns:
        dict: A dictionary containing the prompt template and configurations.
    """
    template = """
    You are an AI assistant for {organization_name}.
    Your role is to answer customer queries using the following context:
    
    {context}

    Question: {query}
    Provide a concise and accurate response.If there are numerical points in the context, then write them as numerical bullet points for good readability. Provide the URL's also if they are 
    available in context. when user said 'hi' or 'hello' greet them. don't start 
    the message with 'hello' or 'response'. If unsure of the answer, respond based 
    on the query. Ensure responses stay relevant to the context and do not include unrelated information.
    
    """

    prompt = PromptTemplate(
        input_variables=["organization_name", "context", "query"],
        template=template,
    )

    return {
        "prompt_template": prompt,
        "configurations": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_documents": num_documents,
            "model": model
        }
    }

def count_tokens_for_url(url):
    """
    Calculate the number of tokens used by the embedding process for a given URL.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        dict: A dictionary with the total token count and a breakdown by page.
    """
    # Fetch content from the URL
    all_content = fetch_all_links_content(url)
    if not all_content:
        return {"error": "No content retrieved from the URL"}

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use OpenAI's tokenizer

    # Token count
    total_tokens = 0
    token_breakdown = []

    for text, metadata in all_content:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        token_breakdown.append({
            "url": metadata.get("url", "Unknown"),
            "tokens_used": len(tokens),
            "title": metadata.get("title", "Untitled")
        })

    return {
        "total_tokens": total_tokens,
        "breakdown": token_breakdown
    }


def get_chatbot_response(query, chat_history, retrieved_context, max_tokens=100, temperature=0.8, num_documents=10, model="gpt-3.5-turbo", organization_name="your organization"):
    """
    Generate a chatbot response using chat history and the given query.

    Args:
        query (str): The user's query.
        chat_history (list): List of dictionaries containing chat history.
        max_tokens (int): Maximum tokens for the response.
        temperature (float): Creativity level for the response.
        num_documents (int): Number of documents for retrieval (if applicable).
        model (str): The model to use for generating responses.
        organization_name (str): Name of the organization for prompt context.

    Returns:
        dict: A dictionary containing the chatbot response and updated chat history.
    """
    # Format the chat history for the prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Define the prompt template
    template = """
    You are an AI assistant for {organization_name}.
    Your role is to answer customer queries using the following context:
    
    {context}

    Question: {query}
    Provide a concise and helpful response. rember the history of the user data also, If unsure about the answer, respond with "I'm not sure about that." or give a reply based on the query
    """

    # Create the LangChain PromptTemplate
    prompt = PromptTemplate(
        input_variables=["organization_name", "context", "query"],
        template=template,
    )

    # Generate the prompt text
    prompt_text = prompt.format(
        organization_name=organization_name,
        context=formatted_history,
        query=query,
        retrieved_context=retrieved_context
    )
    print("---------prompt-text---------:", prompt_text)

    # Call the language model
    from langchain_community.chat_models.openai import ChatOpenAI  # Import the LLM wrapper
    llm = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        model=model
    )

    try:
        llm_response = llm.invoke(prompt_text)  # Generate the response
        response_text = llm_response.content

        # Append the assistant's response to the chat history
        chat_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_history": chat_history
        }
    except Exception as e:
        print(f"Error generating chatbot response: {e}")
        return {
            "error": str(e),
            "chat_history": chat_history
        }

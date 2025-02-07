The `utils.py` file serves as the **core utility and integration module** for the chatbot. It handles:

1. **API Endpoints**:

   - Defines Flask routes like `/chatbot` and `/update-vector-store` for managing user interactions and vector database updates.

2. **Integration**:

   - Combines functionalities from other files:
     - Uses `prompt.py` for prompt generation.
     - Leverages `text_to_doc.py` for text chunking.
     - Utilizes `web_crawler.py` for fetching web data.
     - Connects to the vector store (e.g., ChromaDB).

3. **Data Storage & Retrieval**:
   - Updates and retrieves documents from the vector store.
   - Monitors stored URLs to avoid redundant processing.

<!-- 4. **Chat History Management**:
   - Manages user-specific chat history via sessions for contextual conversations. -->

### Purpose:

It acts as the **main orchestrator**, linking all components and handling API requests to deliver chatbot functionality.

The `text_to_doc.py` file is responsible for:

1. **Text Cleaning**:

   - Removes unnecessary whitespace and formats the text for consistent processing using `clean_text`.

2. **Text Chunking**:

   - Splits large text into smaller chunks using a defined `chunk_size` and `chunk_overlap` via `MarkdownTextSplitter`.

3. **Document Conversion**:

   - Converts each chunk of text into a LangChain `Document` object, embedding metadata for further use.

4. **Output**:
   - Returns a list of `Document` objects, each containing `page_content` (chunked text) and associated `metadata`.

### Purpose:

It prepares raw text data for processing by vector stores or machine learning models, ensuring efficient storage and retrieval.

The `prompt.py` file is responsible for:

1. **Prompt Template Creation**:

   - Defines and formats a reusable prompt template for the chatbot.
   - Includes placeholders for the organization name, context (retrieved documents + chat history), and the user query.

2. **Configuration Management**:

   - Allows customization of parameters like `max_tokens`, `temperature`, `num_documents`, and model selection.

3. **Output**:
   - Returns a dictionary containing:
     - The prompt template.
     - Configuration details for generating chatbot responses.

### Purpose:

It dynamically builds and manages the input prompts used to interact with the language model, ensuring relevant context is included for accurate responses.

The `web_crawler.py` file is responsible for:

1. **Fetching Website Data**:

   - Retrieves HTML content from a given URL using `requests`.

2. **Extracting Relevant Content**:

   - Uses `BeautifulSoup` to parse HTML, remove unnecessary elements (like scripts and styles), and extract meaningful text.
   - Processes metadata such as the page title, description, and keywords.

3. **Fetching All Links**:

   - Crawls all hyperlinks on a main webpage, deduplicates them, and fetches their content.

4. **Monitoring Changes** (if implemented):
   - Detects changes in webpage content for monitoring purposes.

### Purpose:

It gathers and cleans web data, preparing it for further processing, storage, or analysis.

### To run the rag-bot in local:

**Step-1**: create venv : `py -3.10 -m venv .venv`
**step-2**: Activate venv: `.\.venv\scripts\activate`
**Step-3**: Install the requirements.txt files - `pip install -r requirements.txt`
Or `pip install {package-name}`

**step-4**: create a `.env` file and add `OPENAI_API_KEY = {OPENAI_API_KEY}`

**step-5**: run utils.py file - change directory to `chatbot` and in terminal type: `py utils.py`

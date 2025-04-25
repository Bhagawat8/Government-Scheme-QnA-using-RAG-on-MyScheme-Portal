# My-scheme-RAG

## Overview

My-scheme-RAG is an innovative Python and LangChain-powered application that leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, detailed information on Indian government schemes sourced from Myscheme.gov.in; by scraping and indexing consolidated scheme data, the system retrieves the most relevant documents in response to a user query and then employs state-of-the-art LLMs to generate comprehensive answers with context and source snippets, all delivered through a user-friendly Streamlit interface where users simply input their questions and receive clear, authoritative insights into the full range of central and state-level programs.

This project is ideal for citizens, researchers, or policymakers seeking quick and reliable information about government initiatives without navigating multiple websites.

## Features

- **Natural Language Querying**: Ask questions about government schemes in plain language, and receive detailed, context-aware answers.
- **RAG Architecture**: Combines document retrieval with generative AI to ensure accurate and informative responses.
- **Comprehensive Data Processing**: Scrapes and processes detailed scheme data from MyScheme, including eligibility, benefits, and application processes.
- **Persistent Embeddings**: Precomputed embeddings are saved for reuse, reducing setup time in subsequent runs.
- **Efficient Data Storage**: Uses a Chroma vector database for fast and scalable document retrieval.
- **Interactive Streamlit UI**: A web-based interface allows users to input queries, view answers, and inspect source documents.
- **Error Handling**: Includes robust error handling to manage initialization and query processing failures gracefully.

## Data Source and Structure

The application relies on data web scraped from [Myscheme.gov.in](https://www.myscheme.gov.in/), a government platform that provides a centralized repository of scheme information. The data is stored in a CSV file (`cleaned_my_scheme_data_fixed.csv`) and includes the following fields for each scheme:

| Field                        | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| Scheme Name                  | The official name of the government scheme.                                 |
| Eligibility                  | Criteria that applicants must meet to qualify for the scheme.               |
| Details                      | A detailed description of the scheme’s purpose and implementation.          |
| Tags                         | Keywords associated with the scheme (e.g., education, health).              |
| Application Process          | Steps to apply for the scheme.                                             |
| Benefits                     | Advantages or financial assistance provided by the scheme.                  |
| Documents Required           | List of documents needed for application.                                  |
| Source URL                   | Link to the official scheme page on Myscheme.gov.in.                       |
| Ministries/Departments       | Government bodies responsible for the scheme.                              |
| Target Beneficiaries States  | States where the scheme is applicable.                                     |


## RAG Pipeline Structure

The RAG pipeline is the core of the application, seamlessly integrating retrieval and generation to answer user queries. It consists of the following components:

1. **Data Loading**: Reads scheme data from a CSV file and converts each row into a LangChain `Document` object.
2. **Chunking**: Treats each scheme as a single document, eliminating the need for further splitting.
3. **Embedding Generation**: Generates embeddings for each document using the BAAI/bge-m3 model.
4. **Vector Store**: Stores embeddings in a Chroma vector database with cosine similarity for efficient retrieval.
5. **Retrieval**: Uses a custom retriever to embed user queries and fetch the top-k most relevant documents.
6. **Augmentation**: Combines the query with retrieved documents to create a context-rich prompt.
7. **Generation**: Feeds the prompt to the Falcon3-1B-Instruct model to produce a detailed response.
8. **Parsing**: Formats the response for clarity and displays it to the user.

This modular design ensures scalability and maintainability, allowing each component to be updated independently.

## Models

The application employs two key models to power its RAG pipeline:

| Model                     | Role                | Description                                                                 |
|---------------------------|---------------------|-----------------------------------------------------------------------------|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Embedding Model     | A multilingual model supporting over 100 languages, capable of embedding long sequences (up to 8192 tokens). Ideal for capturing detailed scheme descriptions. |
| [tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct) | Large Language Model      | A 3.23B parameter model fine-tuned for instruction following, with an 32K-token context window, suitable for generating coherent and context-aware responses. |


## How It Works

The application processes user queries through a streamlined workflow:

1. **Query Input**: Users enter a question (e.g., “Is there any scheme for disabled students in Kerala?”) via the Streamlit interface.
2. **Embedding Creation**: The query is converted into an embedding using the BAAI/bge-m3 model.
3. **Document Retrieval**: The embedding is compared against the Chroma vector store to retrieve the top 3 most relevant scheme documents (configurable via `top_k`).
4. **Prompt Construction**: The retrieved documents are formatted into a context string, combined with the query and a system prompt that enforces accuracy and markdown formatting.
5. **Response Generation**: The Falcon3-3B-Instruct model processes the prompt to generate a detailed answer, including scheme details like eligibility, benefits, and application steps.
6. **Output Display**: The response is parsed, formatted in markdown, and displayed in the Streamlit UI, alongside source snippets for transparency.

This process ensures that answers are both accurate and grounded in the provided data.

## Tech Stack

The project leverages a robust set of technologies:

| Technology                | Purpose                              |
|---------------------------|--------------------------------------|
| Python                    | Core programming language            |
| LangChain                 | RAG pipeline and retriever framework |
| ChromaDB                  | Vector database for embeddings       |
| Streamlit                 | Web-based user interface             |
| Hugging Face Transformers | Model loading and inference          |
| Pandas                    | Data loading and preprocessing       |
| NumPy                     | Numerical operations for embeddings  |
| PyTorch                   | Model execution and GPU support      |


## Project Structure

The project is organized into modular Python and jupyter files for clarity and maintainability:

| File/Folder                     | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `loader.py`                     | Loads and preprocesses scheme data from CSV into LangChain `Document` objects. |
| `embedding_generator.py`        | Initializes the BAAI/bge-m3 model and generates embeddings for documents.    |
| `vector_store.py`               | Creates and manages the Chroma vector store for storing embeddings.          |
| `retrieval.py`                  | Defines a custom retriever for fetching relevant documents based on queries. |
| `augmentation.py`               | Augments user queries with retrieved context to form model prompts.          |
| `generation.py`                 | Implements the Falcon3-1B-Instruct model for response generation.            |
| `parser.py`                     | Parses and formats model outputs for display.                               |
| `main.py`                       | Orchestrates the RAG pipeline, integrating all components.                   |
| `app.py`                        | Streamlit application for the web interface.                                |
| `chunks_with_embeddings.pkl`    | Pickled file storing documents with precomputed embeddings.                  |
| `chroma_db/`                    | Directory containing the Chroma vector store database.                      |
| `data_cleanining.ipynb`         | Clean the web-scraped datasets: remove HTML tags and convert them into the correct format |
| `prodigalai-rag.ipynb`          | Jupyter notebook files contains complete RAG pipeline for performing new experiments |

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bhagawat8/My-scheme-RAG.git
   cd My-scheme-RAG
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place the `cleaned_my_scheme_data_fixed.csv` file in the project directory or update the `CSV_PATH` in `main.py` to point to its location.

5. **Generate or Load Embeddings**:
   - If embeddings are not precomputed, `main.py` will generate them using the BAAI/bge-m3 model and save them to `chunks_with_embeddings.pkl`.
   - If `chunks_with_embeddings.pkl` exists, it will be loaded to skip embedding generation.

6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```


## Usage

To use the application:

1. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   This starts a local web server, typically at `http://localhost:8501`.

2. **Access the Interface**:
   Open the provided URL in a  [Streamlit](https://streamlit.io/) browser interface.

3. **Query Schemes**:
   - Enter a question about government schemes (e.g., “Financial assistance for disabled students in Kerala”).
   - View the generated answer, formatted in markdown with sections like Scheme Name, Eligibility, Benefits, etc.
   - Inspect source snippets from retrieved documents for verification.

4. **Explore Results**:
   - The interface displays the answer and relevant document excerpts, ensuring transparency and trust in the response.

## Citations

- [Myscheme.gov.in: National Platform for Government Schemes](https://www.myscheme.gov.in/)
- [BAAI/bge-m3: Multilingual Text Embedding Model](https://huggingface.co/BAAI/bge-m3)
- [tiiuae/Falcon3-1B-Instruct: Instruction-Tuned Language Model](https://huggingface.co/tiiuae/Falcon3-1B-Instruct)

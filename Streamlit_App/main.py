import os
import pickle
from loader import load_csv_data
from embedding_generator import initialize_embedding_model, generate_embeddings
from vector_store import create_vector_store
from retrieval import CustomRetriever
from augmentation import augment_query
from generation import FalconGenerator
from parser import CustomOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

CSV_PATH = "cleaned_my_scheme_data_fixed.csv"
EMBEDDINGS_PATH = "chunks_with_embeddings.pkl"

# Check if embeddings already exist
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        chunks_with_embeddings = pickle.load(f)
    print("Loaded embeddings from file.")
else:
    # Load data
    documents = load_csv_data(CSV_PATH)
    
    # Chunk data (in this case, each document is a chunk)
    chunks = documents
    
    # Initialize embedding model
    model = initialize_embedding_model()
    
    # Generate embeddings
    chunks_with_embeddings = generate_embeddings(model, chunks)
    
    # Save embeddings to file
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(chunks_with_embeddings, f)
    print("Generated and saved embeddings.")

# Create or load vector store
collection = create_vector_store(chunks_with_embeddings)
# embedding model
model = initialize_embedding_model()
# Set up retriever
retriever = CustomRetriever(
    collection=collection,  
    embedding_model=model   
)

# Set up generator
generator = FalconGenerator()

# Set up parser
parser = CustomOutputParser()

# Construct RAG chain
rag_chain = RunnableSequence(
    RunnablePassthrough.assign(
        context=lambda x: retriever.retrieve_chunks(x["query"])[0]  # Get chunks only
    ),
    lambda x: augment_query(x["query"], x["context"]),
    lambda messages: generator.generate([messages]),
    lambda result: parser.parse(result.generations[0][0].text)
)

# Example usage
query = {"query": "Is any scheme that provide the Financial Assistance To Disabled Students in kerala"}
final_response = rag_chain.invoke(query)
print(final_response)
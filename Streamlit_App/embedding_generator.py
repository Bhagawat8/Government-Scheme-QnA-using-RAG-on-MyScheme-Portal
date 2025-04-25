import torch
from FlagEmbedding import FlagModel

def initialize_embedding_model():
    """Initialize the embedding model."""
    print("\n=== Initializing Embedding Model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device=device
    )
    return model

def generate_embeddings(model, chunks):
    """Generate embeddings for LangChain Document objects."""
    print("\n=== Generating Embeddings ===")
    
    texts = [doc.page_content for doc in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    
    # Generate embeddings
    embeddings = model.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Add embeddings to document metadata
    for i, doc in enumerate(chunks):
        doc.metadata["embedding"] = embeddings[i]
    
    return chunks
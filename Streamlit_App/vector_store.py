import chromadb

# vector_store
def create_vector_store(chunks_with_embeddings, db_path="chroma_db"):  

    client = chromadb.PersistentClient(path=db_path)

    # 1. Validate input
    if not chunks_with_embeddings:
        raise ValueError("chunks_with_embeddings list cannot be empty")

    # 2. Extract texts, embeddings, metadatas, and ids
    texts = [doc.page_content for doc in chunks_with_embeddings]
    embeddings = [
        # ensure numpy arrays are converted to lists
        doc.metadata["embedding"].tolist()
        for doc in chunks_with_embeddings
    ]
 
    metadatas = [
        # drop the embedding from metadata
        {k: v for k, v in doc.metadata.items() if k != "embedding"}
        for doc in chunks_with_embeddings
    ]
 
    
    ids = [doc.id or str(i) for i, doc in enumerate(chunks_with_embeddings)]

    try:
        collection = client.get_collection(name="rag_data_cosine")
        print("Using existing collection.")
    except:
        # Create a new collection if it doesn't exist
        collection = client.create_collection(
            name="rag_data_cosine",
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection.")
    # 5. Add documents with precomputed embeddings
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )    
    return collection
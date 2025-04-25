from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Tuple, Dict, Any, Optional

# retrieval
class CustomRetriever(BaseRetriever):
    """retriever with FlagModel integration"""
    
    collection: Any  
    embedding_model: Any 
    top_k: int = 3  # top 3 chunks will be retrieve
    similarity_threshold: float = 0.4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Generate query embedding using FlagModel's interface
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Query ChromaDB with proper embedding format
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results with score filtering
        filtered_docs = []
        for doc_content, metadata, distance in zip(results['documents'][0],
                                                 results['metadatas'][0],
                                                 results['distances'][0]):
            score = 1 - distance
            if score >= self.similarity_threshold:
                filtered_docs.append(Document(
                    page_content=doc_content,
                    metadata={**metadata, "score": score}
                ))
        
        if run_manager:
            run_manager.on_retriever_end(
                documents=filtered_docs,
                query=query
            )
        
        print(f"Retrieved {len(filtered_docs)} relevant chunks")
        return filtered_docs

    def retrieve_chunks(self, query: str) -> Tuple[List[str], List[float]]:
        """Public interface method"""
        docs = self.invoke(query)
        return (
            [doc.page_content for doc in docs],
            [doc.metadata.get("score", 0.0) for doc in docs]
        )
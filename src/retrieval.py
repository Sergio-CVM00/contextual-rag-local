"""Hybrid Retrieval - BM25 + Semantic Search"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re


class DualRetriever:
    """Combines BM25 (keyword) and semantic (embedding) search"""
    
    def __init__(self, vector_db, documents: List[Dict]):
        """
        Initialize retriever
        
        Args:
            vector_db: VectorDBManager instance
            documents: List of document dicts with 'document' and 'metadata'
        """
        self.vector_db = vector_db
        self.documents = documents
        
        # Build BM25 index
        self.tokenized_docs = [self._tokenize(doc['document']) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve documents using hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of retrieved document dicts
        """
        # Split results: use semantic for primary, BM25 for diversity
        semantic_k = int(top_k * 0.7)  # 70% from semantic
        bm25_k = top_k - semantic_k     # 30% from BM25
        
        # Semantic search using vector DB
        semantic_results = self._semantic_search(query, semantic_k)
        
        # BM25 search
        bm25_results = self._bm25_search(query, bm25_k)
        
        # Combine and deduplicate
        combined = {}
        
        # Add semantic results (higher priority)
        for doc in semantic_results:
            doc_id = doc.get('id', doc['document'][:50])
            combined[doc_id] = doc
        
        # Add BM25 results if not already present
        for doc in bm25_results:
            doc_id = doc.get('id', doc['document'][:50])
            if doc_id not in combined:
                combined[doc_id] = doc
        
        # Return combined results
        return list(combined.values())[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using semantic similarity"""
        try:
            results = self.vector_db.query(query, n_results=top_k)
            
            retrieved = []
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                retrieved.append({
                    'document': doc,
                    'metadata': results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {},
                    'id': results.get('ids', [[]])[0][i] if results.get('ids') else f"semantic_{i}"
                })
            
            return retrieved
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using BM25"""
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            retrieved = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    retrieved.append({
                        'document': doc['document'],
                        'metadata': doc.get('metadata', {}),
                        'id': f"bm25_{idx}"
                    })
            
            return retrieved
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

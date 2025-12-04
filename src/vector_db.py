"""Chroma DB Vector Database Manager - Local & Free"""

import chromadb
from typing import List, Dict, Optional
from pathlib import Path


class VectorDBManager:
    """Manages local Chroma DB for embeddings (free, no API)"""
    
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma with new persistent client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = None
    
    def create_collection(self, name: str):
        """Create or get a collection"""
        try:
            # Delete if exists
            try:
                self.client.delete_collection(name=name)
            except:
                pass
            
            self.collection = self.client.create_collection(name=name)
            return self.collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_or_create_collection(self, name: str):
        """Get existing collection or create new"""
        try:
            self.collection = self.client.get_or_create_collection(name=name)
            return self.collection
        except Exception as e:
            print(f"Error getting collection: {e}")
            return None
    
    def add_documents(self, 
                     documents: List[str],
                     metadatas: List[Dict],
                     ids: List[str]):
        """Add documents to collection"""
        if not self.collection:
            print("No collection selected")
            return False
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 10) -> Dict:
        """Query the collection"""
        if not self.collection:
            print("No collection selected")
            return {"documents": [], "metadatas": []}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying: {e}")
            return {"documents": [], "metadatas": []}
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents from collection"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get()
            documents = []
            for i, doc in enumerate(results.get("documents", [])):
                documents.append({
                    "document": doc,
                    "metadata": results.get("metadatas", [{}])[i],
                    "id": results.get("ids", [])[i]
                })
            return documents
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(name=name)
            self.collection = None
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def persist(self):
        """Persist collection to disk (automatic with PersistentClient)"""
        # PersistentClient automatically persists, no action needed
        return True

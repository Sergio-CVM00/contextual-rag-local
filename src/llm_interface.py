"""Local Ollama LLM Integration - Zero API costs!"""

import requests
import json
from typing import List, Dict


class LocalLLMInterface:
    """Interface for local Ollama LLM (free, no API costs)"""
    
    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "mistral"):
        self.host = host
        self.model = model
        self.model_url = f"{host}/api/generate"
    
    def generate_context(self, 
                        full_document: str,
                        chunk: str,
                        chunk_index: int) -> str:
        """Generate context using local Ollama LLM (FREE!)"""
        
        prompt = f"""<document>
{full_document[:8000]}
</document>

Here is the chunk to contextualize:

<chunk>
{chunk}
</chunk>

Provide concise context (30-50 words) explaining:
1. Topic/section of this chunk
2. Key entities mentioned
3. Relation to document

Be brief and factual. Context only, no explanations:"""
        
        try:
            response = requests.post(
                self.model_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.0,  # Deterministic
                    "num_predict": 100   # ~50 tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                context = result.get("response", "").strip()
                return context if context else ""
            else:
                print(f"LLM error: {response.status_code}")
                return ""  # Fallback to original chunk
                
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Ollama not running!")
            print("   Run: ollama serve")
            return ""
        except Exception as e:
            print(f"LLM error: {e}")
            return ""
    
    def generate_response(self,
                         query: str,
                         retrieved_chunks: List[Dict]) -> str:
        """Generate response from retrieved chunks (FREE!)"""
        
        # Format context
        context_text = "\n\n".join([
            f"Source {i+1}:\n{c['document'][:300]}"
            for i, c in enumerate(retrieved_chunks[:5])
        ])
        
        prompt = f"""Based on these documents, answer the question:

DOCUMENTS:
{context_text}

QUESTION: {query}

Answer concisely using only information from documents:"""
        
        try:
            response = requests.post(
                self.model_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Slightly creative
                    "num_predict": 500   # Limit response
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "I couldn't generate a response.")
            else:
                return "Error generating response."
                
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Run: ollama serve"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_connection(self) -> bool:
        """Verify Ollama is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

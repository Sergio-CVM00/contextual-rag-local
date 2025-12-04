"""Context Generation - Enriching chunks with local LLM"""

from src.llm_interface import LocalLLMInterface
from typing import List, Tuple


class ContextGenerator:
    """Generate context for chunks using local LLM"""
    
    def __init__(self, llm_interface: LocalLLMInterface = None):
        self.llm = llm_interface or LocalLLMInterface()
    
    def generate_contexts(self, 
                         full_document: str,
                         chunks: List[str],
                         show_progress: bool = True) -> List[str]:
        """
        Generate context for all chunks
        
        Args:
            full_document: Complete document text
            chunks: List of chunks
            show_progress: Show progress (for CLI use)
        
        Returns:
            List of contextualized chunks
        """
        contextualized = []
        
        for i, chunk in enumerate(chunks):
            if show_progress:
                print(f"Contextualizing chunk {i+1}/{len(chunks)}...")
            
            context = self.llm.generate_context(full_document, chunk, i)
            
            # Combine context with chunk
            if context:
                contextualized.append(f"{context}\n\n{chunk}")
            else:
                contextualized.append(chunk)
        
        return contextualized

import os
# Fix tokenizers warning before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import anthropic
from typing import List, Tuple
import json
from datetime import datetime
import config  # Import twojego pliku config.py

class BookRAGSystem:
    def __init__(self, folder_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.folder_path = folder_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []  # Stores info about the source of each chunk
        self.books_info = []
        
        # Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY  # Use the key from config.py
        )
    
    def load_and_process_books(self):
        """Loads and processes all books"""
        print("Loading books...")
        txt_files = [f for f in os.listdir(self.folder_path) if f.endswith(".txt")]
        
        for file_idx, filename in enumerate(txt_files):
            print(f"Processing: {filename}")
            
            # Load and clean text
            with open(os.path.join(self.folder_path, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            clean_text = self.clean_gutenberg_text(raw_text)
            
            # Extract book title (first line after cleaning)
            title = clean_text.split('\n')[0][:100] if clean_text else filename
            
            self.books_info.append({
                'filename': filename,
                'title': title,
                'word_count': len(clean_text.split())
            })
            
            # Split into chunks
            book_chunks = self.chunk_text(clean_text, max_words=400)
            
            # Add metadata for each chunk
            for chunk_idx, chunk in enumerate(book_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'book_index': file_idx,
                    'chunk_index': chunk_idx,
                    'filename': filename,
                    'title': title,
                    'word_count': len(chunk.split())
                })
        
        print(f"Loaded {len(txt_files)} books, {len(self.chunks)} chunks")
        return self
    
    def clean_gutenberg_text(self, text: str) -> str:
        """Cleans text from Gutenberg headers and footers"""
        lines = text.splitlines()
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if line.strip().startswith('*** START OF'):
                start_idx = i + 1
                break
        
        for i, line in enumerate(lines):
            if line.strip().startswith('*** END OF'):
                end_idx = i
                break
        
        cleaned_text = "\n".join(lines[start_idx:end_idx]).strip()
        return cleaned_text
    
    def chunk_text(self, text: str, max_words: int = 400, overlap: int = 50) -> List[str]:
        """Splits text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words - overlap):
            chunk_words = words[i:i + max_words]
            if len(chunk_words) < 50:  # Skip very short chunks
                break
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
        return chunks
    
    def build_index(self):
        """Builds FAISS index"""
        print("Building vector index...")
        
        # Create embeddings with batch processing for better performance
        print("Creating embeddings...")
        embeddings = self.model.encode(
            self.chunks, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            batch_size=32  # Added for better performance
        )
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        
        # Build FAISS index
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_np)
        
        print(f"Index built: {self.index.ntotal} vectors, dimension: {dim}")
        return self
    
    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[dict]]:
        """Searches for the most similar chunks"""
        if self.index is None:
            raise ValueError("Index has not been built! Use build_index()")
        
        # Query embedding
        query_emb = self.model.encode([query], convert_to_tensor=True)
        query_emb_np = query_emb.cpu().numpy().astype('float32')
        
        # Search
        distances, indices = self.index.search(query_emb_np, k)
        
        # Retrieve chunks and metadata
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        retrieved_metadata = [self.chunk_metadata[i] for i in indices[0]]
        
        # Add similarity score
        for i, meta in enumerate(retrieved_metadata):
            meta['similarity_score'] = float(distances[0][i])
        
        return retrieved_chunks, retrieved_metadata
    
    def generate_answer(self, query: str, k: int = 5) -> dict:
        """Generates an answer using RAG"""
        # Search for similar chunks
        chunks, metadata = self.search(query, k)
        
        # Prepare context with source information
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            context_parts.append(f"[Source: {meta['title']}]\n{chunk}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Prepare prompt
        prompt = f"""Based on the following context from various books, answer the question in English.

Context:
{context}

Question: {query}

Instructions:
- Answer precisely based on the provided context
- If possible, indicate which books/sources support your answer
- If the context does not contain enough information, state this clearly
- Answer in English

Answer:"""
        
        # Call Claude
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response.content[0].text
            
        except Exception as e:
            answer = f"Error during answer generation: {str(e)}"
        
        return {
            'query': query,
            'answer': answer,
            'sources': metadata,
            'context_length': len(context),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self) -> dict:
        """Returns system statistics"""
        return {
            'books_count': len(self.books_info),
            'chunks_count': len(self.chunks),
            'books_info': self.books_info,
            'avg_chunk_length': np.mean([len(chunk.split()) for chunk in self.chunks]),
            'total_words': sum([book['word_count'] for book in self.books_info])
        }

# Use system
def main():
    # API from config.py
    
    # Initializing system
    rag = BookRAGSystem("gutenberg_books")
    
    # Create base
    rag.load_and_process_books().build_index()
    
    # Show stats
    stats = rag.get_stats()
    print(f"\nSystem statistics:")
    print(f"Books: {stats['books_count']}")
    print(f"Chunks: {stats['chunks_count']}")
    print(f"Average chunk length: {stats['avg_chunk_length']:.1f} words")
    print(f"Total word count: {stats['total_words']:,}")
    
    print(f"\nLoaded books:")
    for book in stats['books_info']:
        print(f"- {book['title'][:60]}... ({book['word_count']:,} words)")
    
    # Q&A
    print("\n" + "="*60)
    print("System RAG ready! Ask questions (type 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'wyjście']:
            break
        
        if not query:
            continue
        
        print("\nSearching for answer...")
        result = rag.generate_answer(query)
        
        print(f"\nAnswer:")
        print(result['answer'])
        
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['title'][:50]}... (similarity: {source['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
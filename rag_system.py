import os
# Fix tokenizers warning before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import sent_tokenize
import anthropic
from typing import List, Tuple, Dict
import json
from datetime import datetime
import config 
import pickle
from pathlib import Path
import hashlib
from deep_translator import GoogleTranslator
from langdetect import detect

class BookRAGSystem:
    def __init__(self, folder_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.folder_path = folder_path
        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = None
        self.chunks = []
        self.chunk_metadata = []  # Stores info about the source of each chunk
        self.books_info = []
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY  # Use the key from config.py
        )
    
    def semantic_chunk_text(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 400) -> List[str]:
        """Splits text into semantic chunks using sentence boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > max_chunk_size and current_chunk:
                # Join current chunk and add to chunks
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_words
            
            # If we have enough content, create a new chunk
            if current_size >= min_chunk_size and len(current_chunk) > 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def get_cache_path(self, text: str) -> Path:
        """Generate cache path for a text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"

    def get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding or compute new one"""
        cache_path = self.get_cache_path(text)
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        embedding = self.model.encode(text, convert_to_tensor=True).cpu().numpy()
        
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding

    def load_and_process_books(self):
        """Loads and processes all books using semantic chunking"""
        print("Loading books...")
        txt_files = [f for f in os.listdir(self.folder_path) if f.endswith(".txt")]
        
        # Import books info from app.py
        from app import BOOKS_INFO
        
        for file_idx, filename in enumerate(txt_files):
            print(f"Processing: {filename}")
            
            # Load and clean text
            with open(os.path.join(self.folder_path, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            clean_text = self.clean_gutenberg_text(raw_text)
            
            # Get book info from BOOKS_INFO
            book_info = BOOKS_INFO.get(filename, {})
            
            self.books_info.append({
                'filename': filename,
                'title': book_info.get('title', clean_text.split('\n')[0][:100]),
                'author': book_info.get('author', 'Unknown'),
                'year': book_info.get('year', 'Unknown'),
                'word_count': len(clean_text.split())
            })
            
            # Use semantic chunking
            book_chunks = self.semantic_chunk_text(clean_text)
            
            # Add metadata for each chunk
            for chunk_idx, chunk in enumerate(book_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'book_index': file_idx,
                    'chunk_index': chunk_idx,
                    'filename': filename,
                    'title': book_info.get('title', clean_text.split('\n')[0][:100]),
                    'author': book_info.get('author', 'Unknown'),
                    'year': book_info.get('year', 'Unknown'),
                    'word_count': len(chunk.split())
                })
        
        print(f"Loaded {len(txt_files)} books, {len(self.chunks)} chunks")
        return self
    
    def clean_gutenberg_text(self, text: str) -> str:
        """Cleans text from Gutenberg headers and footers"""
        lines = text.splitlines()
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of actual content
        for i, line in enumerate(lines):
            if line.strip().startswith('*** START OF'):
                start_idx = i + 1
                break
        
        # Find end of actual content
        for i, line in enumerate(lines):
            if line.strip().startswith('*** END OF'):
                end_idx = i
                break
        
        # Get the actual content
        content_lines = lines[start_idx:end_idx]
        
        # Remove common Gutenberg artifacts
        cleaned_lines = []
        for line in content_lines:
            # Skip common Gutenberg artifacts
            if any(skip in line.lower() for skip in [
                'produced by', 'html version by', 'transcriber', 
                'proofread by', 'distributed proofreading', 
                'gutenberg', 'etext', 'ebook'
            ]):
                continue
            cleaned_lines.append(line)
        
        cleaned_text = "\n".join(cleaned_lines).strip()
        return cleaned_text
    
    def build_index(self):
        """Builds FAISS index with cached embeddings"""
        print("Building vector index...")
        
        embeddings = []
        for chunk in self.chunks:
            embedding = self.get_cached_embedding(chunk)
            embeddings.append(embedding)
        
        embeddings_np = np.array(embeddings).astype('float32')
        
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_np)
        
        print(f"Index built: {self.index.ntotal} vectors, dimension: {dim}")
        return self
    
    def rerank_results(self, query: str, chunks: List[str], metadata: List[dict], k: int = 5) -> Tuple[List[str], List[dict]]:
        """Rerank results using cross-encoder"""
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk) for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Combine scores with metadata
        scored_results = list(zip(chunks, metadata, scores))
        
        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        # Take top k results
        top_results = scored_results[:k]
        
        # Update metadata with cross-encoder scores
        for _, meta, score in top_results:
            meta['cross_encoder_score'] = float(score)
        
        return [r[0] for r in top_results], [r[1] for r in top_results]

    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[dict]]:
        """Searches for the most similar chunks with reranking"""
        if self.index is None:
            raise ValueError("Index has not been built! Use build_index()")
        
        # Get initial results using FAISS
        query_emb = self.get_cached_embedding(query)
        distances, indices = self.index.search(query_emb.reshape(1, -1), k*2)  # Get more results for reranking
        
        # Retrieve chunks and metadata
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        retrieved_metadata = [self.chunk_metadata[i] for i in indices[0]]
        
        # Add initial similarity scores
        for i, meta in enumerate(retrieved_metadata):
            meta['similarity_score'] = float(distances[0][i])
        
        # Rerank results
        reranked_chunks, reranked_metadata = self.rerank_results(
            query, retrieved_chunks, retrieved_metadata, k
        )
        
        return reranked_chunks, reranked_metadata
    
    def generate_answer(self, query: str, k: int = 5) -> dict:
        """Generates an answer using RAG"""
        # Detect if query is not in English and translate if needed
        original_query = query
        translated_query = query
        detected_lang = 'en'
        
        print(f"\nOriginal query: {original_query}")
        
        try:
            detected_lang = detect(query)
            print(f"Detected language: {detected_lang}")
            
            if detected_lang != 'en':
                translated_query = GoogleTranslator(source=detected_lang, target='en').translate(query)
                print(f"Translated query: {translated_query}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            # Continue with original query if translation fails
            pass

        # Search for similar chunks using translated query
        chunks, metadata = self.search(translated_query, k)
        
        # Prepare context with source information
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            # Get book info from books_info
            book_info = next((book for book in self.books_info if book['filename'] == meta['filename']), None)
            if book_info:
                meta['title'] = book_info['title']
                meta['author'] = book_info.get('author', 'Unknown')
                meta['year'] = book_info.get('year', 'Unknown')
            
            context_parts.append(f"[Source: {meta['title']}]\n{chunk}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Prepare prompt for English answer
        prompt = f"""Based on the following context from various books, answer the question.

Context:
{context}

Question: {translated_query}

Instructions:
- Answer precisely based on the provided context from the books
- Write in a natural, conversational tone
- Make the answer engaging and easy to read
- Use clear formatting with headers and bullet points where appropriate
- Write complete sentences with proper punctuation marks
- Only use bold formatting (**text**) for section headers or truly important concepts

Content Guidelines:
- Focus on practical, actionable advice
- Focus on synthesizing information rather than quoting directly
- Keep any references to source material very brief (max 1-2 sentences)
- If mentioning sources, do it naturally without formal citations

- If the context doesn't contain enough information, state this clearly

Answer:"""
        
        # Call Claude for English answer
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            english_answer = response.content[0].text
            print(f"\nEnglish answer: {english_answer}")
            
            # Translate answer back to original language if needed
            if detected_lang != 'en':
                try:
                    answer = GoogleTranslator(source='en', target=detected_lang).translate(english_answer)
                    print(f"Translated answer: {answer}")
                except Exception as e:
                    print(f"Answer translation error: {str(e)}")
                    answer = english_answer
            else:
                answer = english_answer
            
        except Exception as e:
            answer = f"Error during answer generation: {str(e)}"
        
        return {
            'query': original_query,
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
        
        if query.lower() in ['quit', 'exit', 'end', 'stop', 'q']:
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
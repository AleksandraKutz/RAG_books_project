import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
import config
from rag_system import BookRAGSystem  # Import RAG

app = Flask(__name__)

BOOKS_INFO = {
    "How_to_live_24_hours_a_day.txt": {
        "title": "How to Live on 24 Hours a Day",
        "author": "Arnold Bennett",
        "year": "1910",
        "url": "https://www.gutenberg.org/cache/epub/2274/pg2274.txt",
        "description": "A practical guide to making the most of your time and living a full life."
    },
    "Self_help.txt": {
        "title": "Self-Help",
        "author": "Samuel Smiles", 
        "year": "1859",
        "url": "https://www.gutenberg.org/cache/epub/935/pg935.txt",
        "description": "A classic book on personal development and achieving success through one's own efforts."
    },
    "Science_of_getting_rich.txt": {
        "title": "The Science of Getting Rich",
        "author": "Wallace D. Wattles",
        "year": "1910", 
        "url": "https://www.gutenberg.org/files/65110/65110-0.txt",
        "description": "A philosophy and practical methods for attaining wealth and financial success."
    },
    "Acres_of_Diamonds.txt": {
        "title": "Acres of Diamonds",
        "author": "Russell H. Conwell",
        "year": "1915",
        "url": "https://www.gutenberg.org/cache/epub/368/pg368.txt", 
        "description": "An inspiring story about finding opportunities and wealth in your own surroundings."
    },
    "As_a_man_thinketh": {
        "title": "As a Man Thinketh",
        "author": "James Allen",
        "year": "1903",
        "url": "https://www.gutenberg.org/cache/epub/4507/pg4507.txt",
        "description": "On the power of thought and how mindset influences one's life."
    },
    "The_Principles_of_Scientific_Management.txt": {
        "title": "The Principles of Scientific Management", 
        "author": "Frederick Winslow Taylor",
        "year": "1911",
        "url": "https://www.gutenberg.org/cache/epub/6435/pg6435.txt",
        "description": "The foundations of modern management and optimizing work processes."
    }
}

# Initialize RAG system (global variable)
rag_system = None

def initialize_rag():
    """Initializes the RAG system"""
    global rag_system
    if rag_system is None:
        print("Initializing RAG system...")
        rag_system = BookRAGSystem("gutenberg_books")
        rag_system.load_and_process_books().build_index()
        print("RAG system ready!")
    return rag_system

@app.route('/')
def home():
    """Homepage"""
    return render_template('index.html', books=BOOKS_INFO)

@app.route('/ask', methods=['POST'])
def ask_question():
    """API endpoint for asking questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Ensure the RAG system is initialized
        rag = initialize_rag()
        
        # Generate answer
        result = rag.generate_answer(question, k=5)
        
        # Prepare response
        response = {
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'timestamp': result['timestamp'],
            'stats': {
                'context_length': result['context_length'],
                'sources_count': len(result['sources'])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error during processing: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Returns system statistics"""
    try:
        rag = initialize_rag()
        stats = rag.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/books')
def get_books():
    """Returns information about books"""
    return jsonify(BOOKS_INFO)

if __name__ == '__main__':
    print("🚀 Starting RAG application...")
    print("📚 Books in the system:")
    for filename, info in BOOKS_INFO.items():
        print(f"   - {info['title']} ({info['author']}, {info['year']})")
    
    # Initialize RAG system on startup
    initialize_rag()
    
    print("\nApplication available at: http://localhost:5050")
    app.run(debug=True, host='0.0.0.0', port=5050)
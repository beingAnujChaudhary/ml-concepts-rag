import wikipedia
import json
import os
import re
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Topics to fetch from Wikipedia
WIKI_TOPICS = [
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Support vector machine",
    "Random forest",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Supervised learning",
    "Unsupervised learning",
    "Generative artificial intelligence",
    "Large language model",
    "Transformer (machine learning model)"
]

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
INDEX_NAME = "ml_concepts"

def fetch_wikipedia_pages(topics):
    print("Fetching Wikipedia pages...")
    pages_data = []
    for topic in tqdm(topics):
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            pages_data.append({
                "title": page.title,
                "url": page.url,
                "content": page.content
            })
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
    return pages_data

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks of approximately chunk_size characters with overlap."""
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start next chunk with overlap from end of previous chunk
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk.strip()[-overlap:]
                current_chunk = overlap_text + " " + p + " "
            else:
                current_chunk = p + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def process_and_embed(pages_data):
    print("Chunking text and computing embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = []
    doc_id = 0
    for page in tqdm(pages_data):
        title = page['title']
        url = page['url']
        content = page['content']
        
        # Remove markdown headers and equals signs
        content = re.sub(r'==+ .*? ==+', '', content)
        
        chunks = chunk_text(content)
        
        for chunk in chunks:
            if len(chunk.split()) < 10:
                continue # Skip very short chunks
                
            embedding = model.encode(chunk).tolist()
            
            documents.append({
                "id": str(doc_id),
                "title": title,
                "url": url,
                "text": chunk,
                "embedding": embedding
            })
            doc_id += 1
            
    return documents

def index_to_elasticsearch(documents):
    print(f"Connecting to Elasticsearch at {ELASTIC_URL}...")
    es = Elasticsearch(ELASTIC_URL)
    
    # Define index mapping
    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    # Create index if not exists
    if es.indices.exists(index=INDEX_NAME):
        print(f"Deleting existing index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)
        
    print(f"Creating index '{INDEX_NAME}'...")
    es.indices.create(index=INDEX_NAME, body=mapping)
    
    # Index documents
    print(f"Indexing {len(documents)} documents...")
    for doc in tqdm(documents):
        es.index(index=INDEX_NAME, id=doc["id"], document=doc)
        
    print("Indexing complete.")

if __name__ == "__main__":
    pages_data = fetch_wikipedia_pages(WIKI_TOPICS)
    documents = process_and_embed(pages_data)
    
    # Save a sample to check structure
    with open("sample_docs.json", "w") as f:
        # Save first 5 without full embedding to save space
        sample = [{k: v for k, v in d.items() if k != 'embedding'} for d in documents[:5]]
        json.dump(sample, f, indent=2)
        
    index_to_elasticsearch(documents)

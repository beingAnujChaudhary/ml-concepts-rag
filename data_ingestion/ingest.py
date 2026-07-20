import wikipedia
import json
import os
import re
from pinecone import Pinecone, ServerlessSpec
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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = "ml-concepts"  # Pinecone index names must be lowercase hyphens

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

def index_to_pinecone(documents):
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not found. Please set it in .env")
        return
        
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if not exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(INDEX_NAME)
    
    print(f"Preparing {len(documents)} vectors for Pinecone...")
    vectors = []
    for doc in documents:
        vectors.append({
            "id": doc["id"],
            "values": doc["embedding"],
            "metadata": {
                "title": doc["title"],
                "url": doc["url"],
                "text": doc["text"]
            }
        })
        
    print(f"Upserting to Pinecone...")
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size)):
        index.upsert(vectors=vectors[i:i+batch_size])
        
    print("Indexing complete.")

if __name__ == "__main__":
    pages_data = fetch_wikipedia_pages(WIKI_TOPICS)
    documents = process_and_embed(pages_data)
    
    # Save a sample to check structure
    with open("sample_docs.json", "w") as f:
        # Save first 5 without full embedding to save space
        sample = [{k: v for k, v in d.items() if k != 'embedding'} for d in documents[:5]]
        json.dump(sample, f, indent=2)
        
    index_to_pinecone(documents)

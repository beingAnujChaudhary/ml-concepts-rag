import os
import time

from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = "ml-concepts"

# Dual LLM provider: try OpenAI first, fall back to Groq
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_providers = []
if OPENAI_API_KEY:
    _providers.append({
        "name": "OpenAI",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    })
if GROQ_API_KEY:
    _providers.append({
        "name": "Groq",
        "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
        "model": GROQ_MODEL,
    })


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _llm_complete(messages, temperature=0.0):
    """Try each configured LLM provider in order; return the first successful response."""
    for provider in _providers:
        try:
            response = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{provider['name']}] LLM error: {e}")
    return None

def rewrite_query(user_query):
    """Uses LLM to rewrite the query for better retrieval."""
    prompt = f"""
    You are an expert in Machine Learning. 
    Rewrite the following user query to be more specific and suitable for searching a knowledge base of Wikipedia articles on ML.
    Only output the rewritten query, nothing else.
    
    User query: {user_query}
    """
    
    result = _llm_complete([{"role": "user", "content": prompt}], temperature=0.3)
    return result if result else user_query

def vector_search(query, top_k=5):
    """Performs a vector search on Pinecone."""
    if not PINECONE_API_KEY:
        print("Pinecone API key missing")
        return []
        
    query_vector = embedding_model.encode(query).tolist()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in response['matches']:
            doc = match['metadata']
            doc['id'] = match['id']
            results.append(doc)
            
        return results
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def rerank_results(query, results):
    """Re-rank retrieved documents using a cross-encoder model."""
    if not results:
        return results

    pairs = [[query, doc['text']] for doc in results]
    scores = cross_encoder.predict(pairs)
    for i, score in enumerate(scores):
        results[i]['rerank_score'] = float(score)
    results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    return results

def generate_answer(query, contexts):
    """Generate a grounded explanation and a separate Mermaid concept diagram."""
    context_text = ""
    for doc in contexts:
        context_text += f"Title: {doc['title']}\nText: {doc['text']}\n\n"

    prompt = f"""
    You are a Machine Learning educator. Answer the user's question using ONLY the provided context.
    Explain the concept clearly and concisely for a learner. Do not include a Mermaid diagram in this answer.
    If the context does not contain enough information, say exactly:
    "I don't have enough information to answer that based on my knowledge base."

    Context:
    {context_text}

    Question: {query}

    Answer:
    """

    result = _llm_complete([{"role": "user", "content": prompt}])
    return result if result else "The answer could not be generated. Check the LLM configuration and try again."


def generate_mermaid_diagram(query, answer, contexts):
    """Create valid Mermaid flowchart code grounded in the retrieved context."""
    if not contexts or answer.startswith("I don't have enough information"):
        return ""

    context_text = "\n\n".join(
        f"Title: {doc['title']}\nText: {doc['text']}" for doc in contexts
    )
    prompt = f"""
    Create a compact Mermaid flowchart that illustrates the ML concept in the answer.
    Use ONLY facts supported by the context. Return Mermaid code only, without backticks or commentary.

    Requirements:
    - The first line must be: flowchart TD
    - Use 4 to 8 nodes and directional arrows.
    - Keep labels short, educational, and under 45 characters.
    - Put every node label in double quotes, for example A["Input data"].
    - Use only alphanumeric node IDs.
    - Do not use HTML, markdown, parentheses in labels, click actions, or custom styling.

    Question: {query}
    Answer: {answer}
    Context:
    {context_text}
    """

    diagram = _llm_complete([{"role": "user", "content": prompt}])
    if not diagram:
        return ""
    diagram = diagram.removeprefix("```mermaid").removeprefix("```")
    diagram = diagram.removesuffix("```").strip()
    return diagram if diagram.startswith("flowchart") else ""

def get_rag_response(user_query):
    start_time = time.time()
    
    rewritten_query = rewrite_query(user_query)
    
    search_results = vector_search(rewritten_query, top_k=5)
    reranked_results = rerank_results(rewritten_query, search_results)
    
    # We only use top 3 for the LLM to save context size
    final_contexts = reranked_results[:3]
    
    answer = generate_answer(user_query, final_contexts)
    mermaid_diagram = generate_mermaid_diagram(user_query, answer, final_contexts)

    end_time = time.time()
    response_time_ms = int((end_time - start_time) * 1000)
    
    return {
        "rewritten_query": rewritten_query,
        "contexts": final_contexts,
        "answer": answer,
        "mermaid_diagram": mermaid_diagram,
        "response_time_ms": response_time_ms
    }

import os
import time

from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
INDEX_NAME = "ml_concepts"

es_client = Elasticsearch(ELASTIC_URL)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = OpenAI()

def rewrite_query(user_query):
    """Uses LLM to rewrite the query for better retrieval."""
    prompt = f"""
    You are an expert in Machine Learning. 
    Rewrite the following user query to be more specific and suitable for searching a knowledge base of Wikipedia articles on ML.
    Only output the rewritten query, nothing else.
    
    User query: {user_query}
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return user_query # Fallback to original

def hybrid_search(query, top_k=5):
    """Performs a hybrid search (BM25 + Vector KNN) on Elasticsearch."""
    query_vector = embedding_model.encode(query).tolist()
    
    search_query = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 50,
            "boost": 0.5
        },
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "boost": 0.5
                }
            }
        },
        "size": top_k,
        "_source": ["id", "title", "url", "text"]
    }
    
    try:
        response = es_client.search(index=INDEX_NAME, body=search_query)
        hits = response['hits']['hits']
        
        results = []
        for hit in hits:
            results.append(hit['_source'])
            
        return results
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []

def rerank_results(query, results):
    """
    Placeholder hook for future cross-encoder reranking.

    This currently preserves Elasticsearch ordering and must not be described as
    implemented document reranking in project evaluation claims.
    """
    # Placeholder for actual cross-encoder re-ranking
    # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # pairs = [[query, doc['text']] for doc in results]
    # scores = cross_encoder.predict(pairs)
    # for i in range(len(scores)):
    #     results[i]['score'] = scores[i]
    # results = sorted(results, key=lambda x: x['score'], reverse=True)
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

    try:
        response = llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "The answer could not be generated. Check the OpenAI configuration and try again."


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

    try:
        response = llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        diagram = response.choices[0].message.content.strip()
        diagram = diagram.removeprefix("```mermaid").removeprefix("```")
        diagram = diagram.removesuffix("```").strip()
        return diagram if diagram.startswith("flowchart") else ""
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        return ""

def get_rag_response(user_query):
    start_time = time.time()
    
    rewritten_query = rewrite_query(user_query)
    
    search_results = hybrid_search(rewritten_query, top_k=5)
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

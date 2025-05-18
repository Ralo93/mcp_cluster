import gradio as gr
from textblob import TextBlob
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from smolagents import tool
from typing import TypedDict

class QueryInput(TypedDict):
    query: str

@tool
def predict(query: QueryInput) -> dict:
    """
    Query a PostgreSQL database using a local sentence embedding model and return results as JSON.

    Args:
        query (QueryInput): The textual query to encode and search for in the vector database.

    Returns:
        dict: A dictionary containing a list of the top matching results with metadata, or None if an error occurred.

    Raises:
        Exception: If table verification or query execution fails.

    """
    return query_with_local_model(query)


# Reuse your existing functions
def build_db_url(cfg):
    return f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"

def load_model(model_path="models/embedding_model"):
    model = SentenceTransformer(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def query_similar_vectors(query_embedding, db_url, table="speeches", top_k=5):
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Your table has these columns based on your DataLoader class:
            # id, content, politician, term, position, date, faction, embedding, cluster, topic, topic_desc, cluster_desc
            
            # Modify the query to match your actual table structure
            cur.execute(f"""
                SELECT id, content, politician, term, position, date, faction, 
                       cluster, topic, topic_desc, cluster_desc, 
                       1 - (embedding <#> %s::vector) AS similarity
                FROM {table}
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            results = cur.fetchall()
            return results

from typing import Optional, List, Dict, Any, TypedDict



def query_with_local_model(query: str) -> Optional[Dict[str, Any]]:
    """
    Query a PostgreSQL database using a local sentence embedding model and return results as JSON.

    Args:
        query (str): The textual query to encode and search for in the vector database.

    Returns:
        dict: A dictionary containing a list of the top matching results with metadata, or None if an error occurred.

    Raises:
        Exception: If table verification or query execution fails.

    Requirements:
        - A SentenceTransformer model saved in 'models/embedding_model'.
        - PostgreSQL database with a table named 'speeches' that includes vector embeddings.
        - Defined helper functions: `build_db_url`, `load_model`, `query_similar_vectors`.
    """
    model_path = "models/embedding_model"

    db_config = {
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432",
        "database": "cluster",
        "document_table": "speeches"
    }

    db_url = build_db_url(db_config)
    table_name = db_config["document_table"]

    print("🔄 Loading embedding model...")
    model, _ = load_model(model_path)

    # Verify table structure
    print(f"🔍 Verifying table '{table_name}'...")
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}';
                """)
                columns = cur.fetchall()
                print("Available columns:")
                for col_name, col_type in columns:
                    print(f"  - {col_name} ({col_type})")
    except Exception as e:
        print(f"⚠️ Failed to verify table structure: {str(e)}")

    print("🔄 Encoding query...")
    query_embedding = model.encode([query])[0]

    print("🔎 Searching similar embeddings in database...")
    try:
        results = query_similar_vectors(query_embedding, db_url, table=table_name, top_k=5)

        formatted_results: List[Dict[str, Any]] = []

        for row in results:
            id, content, politician, term, position, date, faction, cluster, topic, topic_desc, cluster_desc, similarity = row

            content_snippet = content[:100] + "..." if len(content) > 100 else content

            formatted_results.append({
                "id": id,
                "politician": politician,
                "faction": faction,
                "term": term,
                "position": position,
                "date": date,
                "topic": topic,
                "topic_desc": topic_desc or "N/A",
                "cluster": cluster,
                "cluster_desc": cluster_desc or "N/A",
                "similarity": round(similarity, 4),
                "content_snippet": content_snippet
            })

        return {
            "query": query,
            "results": formatted_results
        }

    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        print("Please verify your table structure and try again.")
        return {
            "error": str(e),
            "message": "Query execution failed. Please verify your setup."
        }


demo = gr.Interface(fn=predict, inputs="text", outputs="json")


# In your server code:
if __name__ == "__main__":
    demo.launch(mcp_server=True)  # Explicitly set port to 7860
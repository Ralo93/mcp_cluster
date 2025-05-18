import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

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


def main():
    model_path = "models/embedding_model"
    
    # Use the same database config format as in your DataLoader
    db_config = {
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432",
        "database": "cluster",
        "document_table": "speeches"  # Use the same table name as in your DataLoader
    }
    
    db_url = build_db_url(db_config)
    table_name = db_config["document_table"]

    print("🔄 Loading embedding model...")
    model, _ = load_model(model_path)

    # First, verify table structure
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

    while True:
        user_input = input("🔍 Enter your query (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break

        print("🔄 Encoding query...")
        query_embedding = model.encode([user_input])[0]

        print("🔎 Searching similar embeddings in database...")
        try:
            results = query_similar_vectors(query_embedding, db_url, table=table_name, top_k=5)
            print("RESULTS:")
            print(results)

            print("\n📌 Top matches:")
            for i, row in enumerate(results):
                # Format results based on table structure
                id, content, politician, term, position, date, faction, cluster, topic, topic_desc, cluster_desc, similarity = row
                
                # Print a condensed result
                print(f"{i+1}. {politician} ({faction}) - {date} - Similarity: {similarity:.4f}")
                print(f"   Topic: {topic_desc or 'N/A'}")
                
                # Print a snippet of the content (first 100 characters)
                content_snippet = content[:100] + "..." if len(content) > 100 else content
                print(f"   Content: {content_snippet}")
                print()
                
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            print("Please verify your table structure and try again.")

if __name__ == "__main__":
    main()
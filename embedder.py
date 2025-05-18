from sentence_transformers import SentenceTransformer
import psycopg2
from transformers import AutoTokenizer

class Embedder:
    def __init__(self, model_name="thenlper/gte-small", db_url=None):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.db_url = db_url

    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def store_pgvector(self, embeddings, ids, table="embeddings"):
        """Store embeddings in pgvector table"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Create table if not exists
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, vector VECTOR(384));")
                    
                    # Insert embeddings
                    for i, emb in zip(ids, embeddings):
                        cur.execute(f"INSERT INTO {table} (id, vector) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector;", 
                                  (i, emb.tolist()))
                    
                    conn.commit()
                    print(f"✅ Stored {len(embeddings)} embeddings in {table}.")
        except Exception as e:
            print(f"❌ Failed to store embeddings: {str(e)}")

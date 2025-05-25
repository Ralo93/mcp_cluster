from dataclasses import dataclass, field
import json
from typing import Optional, List
import pandas as pd
from sqlalchemy import create_engine, text
import ast  # this is against some sqlalchemy + psy

DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'cluster',
    'document_table': 'speeches'
}


@dataclass
class Speech:
    content: str
    politician: str
    term: int
    position: str
    date: str
    faction: str
    embedding: Optional[List[float]] = None
    cluster: Optional[int] = None
    topic: Optional[int] = None
    topic_desc: Optional[str] = None
    cluster_desc: Optional[str] = None

def build_db_url(cfg):
    return f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"

class DataLoader:
    def __init__(self, csv_path=None, db_config=DB_CONFIG):
        self.csv_path = csv_path
        self.db_config = db_config
        self.db_url = build_db_url(db_config)
        self.engine = create_engine(self.db_url)
        self.df = None

    def load_csv(self):
        """Load and clean CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            if 'faction' in self.df.columns and self.df['faction'].isnull().any():
                self.df['faction'].fillna(self.df['faction'].mode()[0], inplace=True)
            print(f"✅ Loaded CSV with {len(self.df)} rows.")
            return self.df
        except FileNotFoundError:
            print(f"❌ File not found: {self.csv_path}")
            return None
        except Exception as e:
            print(f"❌ Failed to load CSV: {str(e)}")
            return None

    def to_speech_objects(self):
        """Convert DataFrame rows to Speech objects"""
        if self.df is None:
            raise ValueError("❌ DataFrame is empty. Run load_csv() first.")
        
        speech_columns = {
            'speech_content': 'content', 
            'politician_name': 'politician',
            'electoral_term': 'term',
            'position_long': 'position',
            'date': 'date',
            'faction': 'faction'
        }
        
        speeches = []
        for _, row in self.df.iterrows():
            speech_args = {}
            for csv_col, obj_field in speech_columns.items():
                if csv_col in row:
                    value = row[csv_col]
                    # Convert term to int if it exists
                    if csv_col == 'electoral_term':
                        value = int(value) if pd.notna(value) else 0
                    speech_args[obj_field] = value
                else:
                    # Use empty values for missing columns
                    speech_args[obj_field] = "" if obj_field != 'term' else 0
            
            speeches.append(Speech(**speech_args))
            
        print(f"✅ Created {len(speeches)} Speech objects.")
        return speeches

    def check_pgvector_extension(self):
        """Check if pgvector extension is available and install it if possible"""
        try:
            with self.engine.connect() as conn:
                # Check if vector extension is available
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'"
                )).scalar()
                
                if result == 0:
                    print("⚠️ pgvector extension is not available in this PostgreSQL installation.")
                    print("   Please install it first: https://github.com/pgvector/pgvector#installation")
                    return False
                
                # Try to create the extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("✅ pgvector extension is available and enabled.")
                return True
        except Exception as e:
            print(f"⚠️ Failed to check/create pgvector extension: {str(e)}")
            return False
            
 # Add this to your data_loader.py file or equivalent

    def save_to_db_in_batches(self, speeches, batch_size=500):
        """Save speeches to database in batches with proper type conversion"""
        if not speeches:
            print("⚠️ No speeches to save.")
            return 0
            
        table = self.db_config.get("document_table", "speeches")
        
        # Check if pgvector is available
        vector_available = self.check_pgvector_extension()
        
        # Adjust table creation SQL based on vector availability
        if vector_available:
            print("PGVECTOR working!!")
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                politician TEXT,
                term INTEGER,
                position TEXT,
                date TEXT,
                faction TEXT,
                embedding VECTOR(768), 
                cluster INTEGER,
                topic INTEGER,
                topic_desc TEXT,
                cluster_desc TEXT
            );
            """
        else:
            print("PGVECTOR not working!!")
            # Alternative: store embeddings as JSON array if pgvector not available
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                politician TEXT,
                term INTEGER,
                position TEXT,
                date TEXT,
                faction TEXT,
                embedding_json JSON,
                cluster INTEGER,
                topic INTEGER,
                topic_desc TEXT,
                cluster_desc TEXT
            );
            """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(create_table_sql))
                print(f"📋 Table '{table}' is ready.")

                # Adjust insert SQL based on vector availability
                if vector_available:
                    insert_sql = f"""
                    INSERT INTO {table} (
                        content, politician, term, position, date, faction, 
                        embedding, cluster, topic, topic_desc, cluster_desc
                    ) VALUES (
                        :content, :politician, :term, :position, :date, :faction,
                        :embedding, :cluster, :topic, :topic_desc, :cluster_desc
                    );
                    """
                else:
                    insert_sql = f"""
                    INSERT INTO {table} (
                        content, politician, term, position, date, faction, 
                        embedding_json, cluster, topic, topic_desc, cluster_desc
                    ) VALUES (
                        :content, :politician, :term, :position, :date, :faction,
                        :embedding_json, :cluster, :topic, :topic_desc, :cluster_desc
                    );
                    """

                total_inserted = 0
                for i in range(0, len(speeches), batch_size):
                    batch = speeches[i:i+batch_size]
                    
                    if vector_available:
                        values = []
                        for s in batch:
                            # Convert numpy types to Python native types
                            speech_dict = {
                                "content": s.content,
                                "politician": s.politician,
                                "term": int(s.term) if hasattr(s.term, 'item') else s.term,
                                "position": s.position,
                                "date": s.date,
                                "faction": s.faction,
                                "embedding": s.embedding,
                                "cluster": int(s.cluster) if hasattr(s.cluster, 'item') else s.cluster,
                                "topic": int(s.topic) if hasattr(s.topic, 'item') else s.topic,
                                "topic_desc": s.topic_desc,
                                "cluster_desc": s.cluster_desc
                            }
                            values.append(speech_dict)
                    else:
                        # Store as JSON if vector not available
                        import json
                        values = []
                        for s in batch:
                            speech_dict = {
                                "content": s.content,
                                "politician": s.politician,
                                "term": int(s.term) if hasattr(s.term, 'item') else s.term,
                                "position": s.position,
                                "date": s.date,
                                "faction": s.faction,
                                "embedding_json": json.dumps(s.embedding) if s.embedding else None,
                                "cluster": int(s.cluster) if hasattr(s.cluster, 'item') else s.cluster,
                                "topic": int(s.topic) if hasattr(s.topic, 'item') else s.topic,
                                "topic_desc": s.topic_desc,
                                "cluster_desc": s.cluster_desc
                            }
                            values.append(speech_dict)

                    conn.execute(text(insert_sql), values)
                    print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} rows.")
                    total_inserted += len(batch)

            print(f"🎉 Done! Inserted {total_inserted} speeches into '{table}'.")
            return total_inserted
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0
            
    def load_from_db(self, table=None):
        """Load speeches from database"""
        if table is None:
            table = self.db_config.get("document_table", "speeches")
            
        try:
            # Check if pgvector is enabled
            vector_available = self.check_pgvector_extension()
            
            # Adjust query based on vector availability
            if vector_available:
                query = f"SELECT * FROM {table}"
            else:
                query = f"SELECT id, content, politician, term, position, date, faction, embedding_json, cluster, topic, topic_desc, cluster_desc FROM {table}"
                
            # Load data
            df = pd.read_sql(query, self.engine)
            
            # Convert embeddings from JSON if needed
            if not vector_available and 'embedding_json' in df.columns:
                df['embedding'] = df['embedding_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else None)
            
            print(f"✅ Loaded {len(df)} speeches from database.")


            # After loading df to check against sqlalchemy and psyogps2 compatibility problems
            if 'embedding' in df.columns:
                def parse_vector(e):
                    if isinstance(e, str):
                        try:
                            return ast.literal_eval(e)
                        except Exception:
                            return None
                    return e

                df['embedding'] = df['embedding'].apply(parse_vector)

            return df
        except Exception as e:
            print(f"❌ Failed to load speeches from database: {str(e)}")
            return None

import json
import os

import numpy as np
from clusterer import Clusterer
from dashboard import Dashboard
from data_loader import DataLoader, Speech, build_db_url
from embedder import Embedder
from topic_modeler import TopicModeler
from umap import UMAP
from hdbscan import HDBSCAN
import joblib

LENGTH_DESC = 10 # no need to also change it in topic_modeler
MIN_CLUSTER_SIZE = 100

def main():
    # === Configuration ===
    CSV_PATH = "data/leg20.csv"
    DB_CONFIG = {
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432',
        'database': 'cluster',
        'document_table': 'speeches'
    }
    DB_URL = build_db_url(DB_CONFIG)

    # === Flags for processing steps ===
    LOAD_FROM_DB = False  # Set to True to load from DB instead of CSV
    SKIP_EMBEDDING = False  # Set to True to skip embedding generation
    SKIP_CLUSTERING = False  # Set to True to skip clustering
    SKIP_TOPIC_MODELING = False  # Set to True to skip topic modeling
    #SKIP_DASHBOARD = False  # Set to True to skip dashboard
    MODEL_DIR = "models"
            
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # === Load and Prepare ===
        loader = DataLoader(CSV_PATH, DB_CONFIG)
        
        # Check if we should load from database or CSV
        if LOAD_FROM_DB:
            print("🔄 Loading speeches from database...")
            speeches_df = loader.load_from_db()
            if speeches_df is None or len(speeches_df) == 0:
                print("🚨 No speeches found in database. Loading from CSV...")
                LOAD_FROM_DB = False  # Fall back to CSV
            else:
                # Convert DataFrame to Speech objects
                speech_objs = []
                for _, row in speeches_df.iterrows():
                    speech = Speech(
                        content=row['content'],
                        politician=row['politician'],
                        term=row['term'],
                        position=row['position'],
                        date=row['date'],
                        faction=row['faction']
                    )
                    
                    # Add optional fields if available
                    if 'embedding' in row and row['embedding'] is not None:
                        speech.embedding = row['embedding']
                    elif 'embedding_json' in row and row['embedding_json'] is not None:
                        speech.embedding = json.loads(row['embedding_json'])
                        
                    if 'cluster' in row and row['cluster'] is not None:
                        speech.cluster = row['cluster']
                    if 'topic' in row and row['topic'] is not None:
                        speech.topic = row['topic']
                    if 'topic_desc' in row and row['topic_desc'] is not None:
                        speech.topic_desc = row['topic_desc']
                    if 'cluster_desc' in row and row['cluster_desc'] is not None:
                        speech.cluster_desc = row['cluster_desc']
                        
                    speech_objs.append(speech)
                
                print(f"✅ Converted {len(speech_objs)} speeches from database.")
        
        # If not loading from DB or DB load failed, load from CSV
        if not LOAD_FROM_DB:
            df = loader.load_csv()
            if df is None:
                print("🚨 No data loaded. Check CSV path.")
                return
                
            speech_objs = loader.to_speech_objects()
        
        texts = [s.content for s in speech_objs]
        
        # Check if all speeches have embeddings already
        has_embeddings = all(s.embedding is not None for s in speech_objs)
        
        # === Embedding ===
        if not SKIP_EMBEDDING:
            print("🔄 Generating embeddings...")

            #TODO Changed Embedding Model: intfloat/multilingual-e5-large
            embedder = Embedder(db_url=DB_URL)#, model_name="intfloat/multilingual-e5-large")
            embeddings = embedder.encode(texts)
            
            # Assign embeddings to speech objects
            for i, s in enumerate(speech_objs):
                s.embedding = embeddings[i].tolist()
                
            embedder.model.save_pretrained(os.path.join(MODEL_DIR, "embedding_model"))
            embedder.tokenizer.save_pretrained(os.path.join(MODEL_DIR, "embedding_model"))
            print("EMBEDDING MODEL SAVED")
            #else:
             #   print("⚠️ Embedding model is not a HuggingFace model and cannot be saved this way.")
        else:
            print("✅ Using existing embeddings.")
            embedder = Embedder(db_url=DB_URL)
            # Convert to numpy array if needed
            embeddings = np.array([s.embedding for s in speech_objs])
        
        # === Clustering ===
        has_clusters = all(s.cluster is not None for s in speech_objs)
        #print(has_clusters)
        
        if not SKIP_CLUSTERING:
            print("🔄 Clustering speeches...")
            clusterer = Clusterer(min_cluster_size=MIN_CLUSTER_SIZE)  # Adjust min_cluster_size based on your data size
            reduced = clusterer.reduce(embeddings) # THis id the reduced_2d_projection
            clusters = clusterer.cluster(reduced)
            
            # Assign clusters to speech objects
            for i, s in enumerate(speech_objs):
                s.cluster = int(clusters[i])
                s.cluster_desc = f"Cluster {clusters[i]}"



        else:
            print("✅ Skipping clustering.")
            clusterer = Clusterer()
            # Just reduce for visualization
            #reduced = clusterer.reduce(embeddings)
            #reduced_2d = UMAP(n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            #joblib.dump(reduced_2d, os.path.join(MODEL_DIR, "reduced_2d_projection.pkl"))


        # === Topic Modeling ===
        has_topics = all(s.topic is not None for s in speech_objs)

        if not SKIP_TOPIC_MODELING:
            print("🔄 Generating topic model...")
            topic_modeler = TopicModeler(embedder.model, clusterer.umap, clusterer.hdbscan)
            tm = topic_modeler.fit(texts, embeddings)
            
            # Use the new transform method that handles missing prediction data
            try:
                topics, _ = topic_modeler.transform(texts, embeddings)
                
                # Assign topics to speech objectss
                for i, s in enumerate(speech_objs):
                    s.topic = topics[i]
                    topic_words = tm.get_topic(s.topic)
                    s.topic_desc = ", ".join([w[0] for w in topic_words[:LENGTH_DESC]]) if topic_words else "No topic"
            except Exception as e:
                print(f"⚠️ Topic assignment failed: {str(e)}")
                print("⚠️ Continuing without topic assignments")
        else:
            print("✅ Using existing topics.")
            # Initialize topic modeler without fitting
            topic_modeler = TopicModeler(embedder.model, clusterer.umap, clusterer.hdbscan)
            # We need to fit with existing data to have a model for visualization
            tm = topic_modeler.fit(texts, embeddings)
        
        # Save topic model
        joblib.dump(tm, os.path.join(MODEL_DIR, "topic_model.pkl"))

        # === Save all processed data to database ===
        print("🔄 Saving processed data to database...")
        loader.save_to_db_in_batches(speech_objs)
        
            
    except Exception as e:
        print(f"🚨 Error in main process: {str(e)}")
        import traceback
        traceback.print_exc()

    print("🏁 Process completed.")



if __name__ == "__main__":

    # Example usage (for reference, not part of the class):
    main()




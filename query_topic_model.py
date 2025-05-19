import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import pickle
from bertopic import BERTopic
from datetime import datetime
import matplotlib.pyplot as plt


# ---------------------- CONFIG ---------------------- #

DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'cluster',
    'document_table': 'speeches'
}

def build_db_url(config):
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

MODEL_PATH = "models/topic_model.pkl"

# ---------------------- DATA LOADING ---------------------- #

from bertopic import BERTopic

def load_model(path=MODEL_PATH):
    return BERTopic.load(path)


def load_data_from_db():
    db_url = build_db_url(DB_CONFIG)
    engine = create_engine(db_url)
    
    query = f"""
        SELECT content, politician, term, position, date, faction
        FROM {DB_CONFIG['document_table']}
        WHERE content IS NOT NULL AND date IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ---------------------- VISUALIZATION FUNCTIONS ---------------------- #

def show_topic_info(topic_model):
    print("\n📌 Topics Overview")
    print(topic_model.get_topic_info())

def show_hierarchy(topic_model):
    print("\n📊 Showing topic hierarchy...")
    topic_model.visualize_hierarchy().show()

def show_topics_over_time(topic_model, docs, timestamps):
    print("\n📈 Topics over time...")
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time).show()

def show_topics_per_class(topic_model, docs, classes):
    print("\n📎 Topics per class (e.g., political faction)...")
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    print(topics_per_class.head(10))
    topic_model.visualize_topics_per_class(topics_per_class).show()

def show_similarity(topic_model):
    print("\n🔗 Topic similarity matrix...")
    topic_model.visualize_heatmap().show()

def show_barchart(topic_model):
    print("\n📊 Top topics as barchart...")
    topic_model.visualize_barchart().show()

# ---------------------- MAIN ---------------------- #

def main():
    topic_model = load_model()
    df = load_data_from_db()

    docs = df["content"].tolist()
    timestamps = df["date"].tolist()
    factions = df["faction"].tolist()

    show_topic_info(topic_model)
    #show_barchart(topic_model)
    #show_similarity(topic_model)
    #show_hierarchy(topic_model)
    #show_topics_over_time(topic_model, docs, timestamps)
    show_topics_per_class(topic_model, docs, factions)

if __name__ == "__main__":
    main()

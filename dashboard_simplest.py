# dashboard.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from sqlalchemy import create_engine
import traceback
import os
from umap import UMAP
import colorsys
from bertopic import BERTopic
MODEL_PATH = "models/topic_model.pkl"

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
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_url = build_db_url(db_config)
        self.engine = create_engine(self.db_url)
    
    def load_speeches(self):
        """Load speeches from database"""
        table = self.db_config.get("document_table", "speeches")
        
        try:
            # Load speeches from database
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, self.engine)
            
            if len(df) == 0:
                return []
                
            print(f"✅ Loaded {len(df)} speeches from database")
            
            # Convert to Speech objects
            speeches = []
            for _, row in df.iterrows():
                speech = Speech(
                    content=row['content'],
                    politician=row['politician'],
                    term=int(row['term']) if hasattr(row['term'], 'item') else row['term'],
                    position=row['position'],
                    date=row['date'],
                    faction=row['faction']
                )
                
                # Process embedding - ensure it's a list of floats
                embedding = None
                
                if 'embedding' in row and row['embedding'] is not None:
                    try:
                        # If it's a string (JSON), parse it
                        if isinstance(row['embedding'], str):
                            embedding = json.loads(row['embedding'])
                        # If it's already an iterable, convert to list
                        elif hasattr(row['embedding'], '__iter__'):
                            embedding = list(row['embedding'])
                    except:
                        pass
                # Check embedding_json field as fallback
                elif 'embedding_json' in row and row['embedding_json'] is not None:
                    try:
                        embedding = json.loads(row['embedding_json'])
                    except:
                        pass
                
                # Verify embedding is valid (list of floats)
                if embedding is not None:
                    try:
                        # Convert all elements to float and verify
                        embedding = [float(x) for x in embedding]
                        speech.embedding = embedding
                    except:
                        print(f"⚠️ Invalid embedding format for speech by {speech.politician}")
                        speech.embedding = None
                
                # Handle other fields with type conversion
                for field, attr in [('cluster', 'cluster'), ('topic', 'topic'), 
                                ('topic_desc', 'topic_desc'), ('cluster_desc', 'cluster_desc')]:
                    if field in row and row[field] is not None:
                        # Convert numbers to Python int
                        if field in ['cluster', 'topic'] and hasattr(row[field], 'item'):
                            setattr(speech, attr, int(row[field]))
                        else:
                            setattr(speech, attr, row[field])
                    
                speeches.append(speech)
            
            return speeches
            
        except Exception as e:
            st.error(f"Failed to load speeches from database: {str(e)}")
            traceback.print_exc()
            return []

def generate_2d_embeddings(speeches, force_recalculate=False):
    """Generate or load 2D embeddings for visualization"""
    cache_file = "models/embeddings_2d_cache.npy"
    
    # Try to load from cache
    if os.path.exists(cache_file) and not force_recalculate:
        try:
            embeddings_2d = np.load(cache_file)
            if len(embeddings_2d) == len(speeches):
                print(f"✅ Loaded cached 2D embeddings from models directory!")
                return embeddings_2d
        except:
            pass
    
    # Generate new embeddings
    print("🔄 Generating 2D embeddings...")
    
    # Extract embeddings from speech objects and ensure they are valid
    valid_embeddings = []
    valid_indices = []
    
    for i, speech in enumerate(speeches):
        if speech.embedding is not None:
            # Ensure embedding is a flat list of floats with consistent length
            try:
                # First check if it's already a numpy array
                if isinstance(speech.embedding, np.ndarray):
                    emb = speech.embedding
                else:
                    # Convert to list of floats
                    emb = [float(x) for x in speech.embedding]
                
                # Verify length
                if len(emb) > 0:  # Make sure it's not empty
                    valid_embeddings.append(emb)
                    valid_indices.append(i)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Invalid embedding format for speech {i}: {e}")
    
    if not valid_embeddings:
        print("❌ No valid embeddings found in speeches")
        return np.zeros((len(speeches), 2))
    
    print(f"Using {len(valid_embeddings)} valid embeddings out of {len(speeches)} speeches")
    
    # Verify all embeddings have the same dimension
    emb_dim = len(valid_embeddings[0])
    uniform_embeddings = []
    uniform_indices = []
    
    for i, emb in enumerate(valid_embeddings):
        if len(emb) == emb_dim:
            uniform_embeddings.append(emb)
            uniform_indices.append(valid_indices[i])
        else:
            print(f"⚠️ Skipping embedding with inconsistent dimension: {len(emb)} (expected {emb_dim})")
    
    if not uniform_embeddings:
        print("❌ No embeddings with consistent dimensions found")
        return np.zeros((len(speeches), 2))
    
    # Convert to numpy array
    embeddings_array = np.array(uniform_embeddings)
    
    # Generate 2D projection
    reducer = UMAP(n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    
    try:
        reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        # Create a full-sized embedding array initialized with zeros
        full_embeddings_2d = np.zeros((len(speeches), 2))
        
        # Fill in the valid embeddings
        for i, idx in enumerate(uniform_indices):
            if i < len(reduced_embeddings):  # Safety check
                full_embeddings_2d[idx] = reduced_embeddings[i]
        
        # Save to cache
        np.save(cache_file, full_embeddings_2d)
        print(f"✅ Generated 2D embeddings for {len(reduced_embeddings)} speeches")
        
        return full_embeddings_2d
    except Exception as e:
        print(f"❌ Error reducing dimensions: {str(e)}")
        traceback.print_exc()
        return np.zeros((len(speeches), 2))

def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        # Use HSV color space to generate evenly spaced hues
        hue = i / n
        # Keep saturation and value high for good visibility
        saturation = 0.7
        value = 0.9
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to hex format
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

class Dashboard:
    def __init__(self, embeddings_2d, speeches):
        self.embeddings_2d = embeddings_2d
        self.speeches = speeches
        # Political party colors
        self.party_colors = {
            "SPD": "#E3000F",       # Red
            "CDU/CSU": "#000000",   # Black
            "Bündnis 90/Die Grünen": "#1AA037", # Green
            "GRÜNE": "#1AA037",     # Green
            "FDP": "#FFED00",       # Yellow
            "AfD": "#0489DB",       # Blue
            "DIE LINKE.": "#BE3075" # Magenta
        }
        
        # Generate cluster colors dynamically
        self.cluster_colors = {}
        
    
    def run(self):
        # Set wide layout
        st.set_page_config(layout="wide", page_title="Speech Visualization Dashboard")
        
        # Main title and introduction
        st.title("Political Speech Visualization Dashboard")
        st.markdown("""
        This dashboard allows you to explore political speeches using embeddings visualization. 
        Speeches are clustered based on semantic similarity, allowing you to identify common themes and patterns.
        """)
        
        # Generate cluster colors if needed
        unique_clusters = sorted(set(s.cluster for s in self.speeches if s.cluster is not None))
        if not self.cluster_colors or len(self.cluster_colors) != len(unique_clusters):
            cluster_colors = generate_distinct_colors(len(unique_clusters))
            self.cluster_colors = {cluster_id: color for cluster_id, color in zip(unique_clusters, cluster_colors)}
        
        # Create sidebar for controls
        with st.sidebar:
            st.header("Visualization Controls")
            
            # Color scheme selection - store in session state to share between sidebar and tabs
            if 'color_by' not in st.session_state:
                st.session_state.color_by = "Cluster"  # Default value
            
            color_by_sidebar = st.radio(
                "Color by:", 
                ["Faction", "Cluster"], 
                index=0 if st.session_state.color_by == "Faction" else 1,
                horizontal=False,
                key="sidebar_color_by"
            )
            # Update session state when sidebar control changes
            st.session_state.color_by = color_by_sidebar
            
            # Marker size and opacity
            #marker_size = st.slider("Marker Size", min_value=5, max_value=20, value=10, key="sidebar_marker_size")
            marker_opacity = st.slider("Marker Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1, key="sidebar_marker_opacity")
            
            # Advanced settings
            with st.expander("Advanced Settings", expanded=False):
                force_recalculate = st.checkbox("Force recalculate embeddings", value=False)
                show_cluster_labels = st.checkbox("Show cluster labels", value=True)
        
        # Main content area - Split into tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Visualization", "Cluster Analysis", "Data Table", "Speech Contents"])
        
        # Prepare data for the visualization
        data = {
            'x': self.embeddings_2d[:, 0],
            'y': self.embeddings_2d[:, 1],
            'cluster': [s.cluster for s in self.speeches],
            'topic': [s.topic for s in self.speeches],
            'topic_desc': [s.topic_desc for s in self.speeches],
            'cluster_desc': [s.cluster_desc for s in self.speeches],
            'faction': [s.faction for s in self.speeches],
            'politician': [s.politician for s in self.speeches],
            'position': [s.position for s in self.speeches],
            'date': [s.date for s in self.speeches],
            'content_preview': [s.content[:100] + "..." if len(s.content) > 100 else s.content for s in self.speeches],
            'speech_index': list(range(len(self.speeches)))  # Add index for selection
        }
        
        df = pd.DataFrame(data)
        
        # Get the current color_by setting from session state
        color_by = st.session_state.color_by
        
        with tab1:
            topic_model = load_model()

            topic_info = topic_model.get_topic_info()
            topic_descriptions = topic_info[topic_info.Topic != -1]["Name"].tolist()

            # Build color map using topic_desc as key
            topic_colors = px.colors.qualitative.Alphabet
            topic_color_map = {
                desc: topic_colors[i % len(topic_colors)]
                for i, desc in enumerate(topic_descriptions)
            }

            # Optional: Assign black to outliers
            topic_color_map["-1"] = "#000000"

            # Prepare data from self.speeches and 2D projection
            data = {
                'x': self.embeddings_2d[:, 0],
                'y': self.embeddings_2d[:, 1],
                'cluster': [s.cluster for s in self.speeches],
                'topic': [s.topic for s in self.speeches],
                'topic_desc': [s.topic_desc for s in self.speeches],
                'cluster_desc': [s.cluster_desc for s in self.speeches],
                'faction': [s.faction for s in self.speeches],
                'politician': [s.politician for s in self.speeches],
                'position': [s.position for s in self.speeches],
                'date': [s.date for s in self.speeches],
                'content_preview': [
                    s.content[:100] + "..." if len(s.content) > 100 else s.content
                    for s in self.speeches
                ],
                'content_full': [
                    s.content[:] for s in self.speeches
                ],
                'speech_index': list(range(len(self.speeches)))
            }
            df = pd.DataFrame(data)

            #st.subheader("Visualization Controls")

           
            control_col1, control_col2 = st.columns([2, 1])

            with control_col1:
                color_mode = st.radio(
                    "View Mode",
                    ["Topic View", "Faction View"],
                    key="viz_color_mode",
                    horizontal=True
                )


            if color_mode == "Faction View":
                fig = px.scatter(
                    df, x="x", y="y",
                    color="faction",
                    hover_data=["politician", "date", "topic_desc"],
                    custom_data=["speech_index", "content_preview", "topic", "topic_desc", "cluster_desc"],
                    color_discrete_map=self.party_colors,
                    height=700
                )

                fig.update_traces(marker=dict(size=13, opacity=0.8))
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.subheader("🧠 Speech Map by Topic")

                fig = px.scatter(
                    df, x="x", y="y",
                    color="topic_desc",
                    color_discrete_map=topic_color_map,
                    hover_data=["politician", "faction", "date", "topic_desc"],
                    custom_data=["speech_index", "content_preview", "topic", "topic_desc", "cluster_desc"],
                    height=700
                )

                fig.update_traces(marker=dict(size=13, opacity=0.8))
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

                st.plotly_chart(fig, use_container_width=True)


            #st.subheader("📌 Topic Overview")
            #st.plotly_chart(topic_model.visualize_barchart(top_n_topics=8), use_container_width=True)
  
            # Button to trigger loading of faction insights
            if st.button("🔍 Load Faction Insights"):
               
                st.subheader("📎 Topics by Faction")
                topics_per_class = topic_model.topics_per_class(
                    docs=df["content_full"],  # Or [s.content for s in self.speeches]
                    classes=df["faction"]
                )
                st.plotly_chart(topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=None, custom_labels=True), use_container_width=True)
            else:
                st.info("Click the button above to load topic statistics by faction.")

        

# Main function
if __name__ == "__main__":
    # Database configuration
    DB_CONFIG = {
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432',
        'database': 'cluster',
        'document_table': 'speeches'
    }
    
    try:
        # Load data
        loader = DataLoader(DB_CONFIG)
        speeches = loader.load_speeches()
        
        if not speeches:
            st.error("No speeches found in database")
            st.stop()
        
        # Check if advanced settings panel exists to get force_recalculate value
        if 'force_recalculate' not in st.session_state:
            st.session_state.force_recalculate = False
        
        # Generate embeddings
        embeddings_2d = generate_2d_embeddings(speeches, st.session_state.force_recalculate)
        
        # Run dashboard
        dashboard = Dashboard(embeddings_2d, speeches)
        dashboard.run()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        traceback.print_exc()
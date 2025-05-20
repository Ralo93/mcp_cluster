# dashboard.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import json
import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from sqlalchemy import create_engine, text
import traceback
from bertopic import BERTopic

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

class DBLoader:
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_url = build_db_url(db_config)
        self.engine = create_engine(self.db_url)
    
    def load_speeches(self):
        """Load speeches from database"""
        table = self.db_config.get("document_table", "speeches")
        
        try:
            # Try to load with pgvector first
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, self.engine)
            
            # If embedding column exists and is pgvector
            if 'embedding' in df.columns:
                # Convert pgvector to list if necessary
                print(f"✅ Loaded {len(df)} speeches with pgvector embeddings.")
            # If no embedding column but embedding_json exists
            elif 'embedding_json' in df.columns:
                df['embedding'] = df['embedding_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else None)
                print(f"✅ Loaded {len(df)} speeches with JSON embeddings.")
            else:
                print(f"⚠️ Loaded {len(df)} speeches but no embeddings found.")
            
            # Convert to Speech objects
            speeches = []
            for _, row in df.iterrows():
                speech = Speech(
                    content=row['content'],
                    politician=row['politician'],
                    term=row['term'],
                    position=row['position'],
                    date=row['date'],
                    faction=row['faction']
                )
                
                # Add optional fields if available
                if 'embedding' in row and hasattr(row['embedding'], '__iter__'):
                    # Convert numpy array or other iterable to list
                    speech.embedding = list(row['embedding'])
                elif 'embedding_json' in row and row['embedding_json'] is not None:
                    try:
                        speech.embedding = json.loads(row['embedding_json'])
                    except:
                        pass
                        
                if 'cluster' in row and row['cluster'] is not None:
                    speech.cluster = row['cluster']
                if 'topic' in row and row['topic'] is not None:
                    speech.topic = row['topic']
                if 'topic_desc' in row and row['topic_desc'] is not None:
                    speech.topic_desc = row['topic_desc']
                if 'cluster_desc' in row and row['cluster_desc'] is not None:
                    speech.cluster_desc = row['cluster_desc']
                    
                speeches.append(speech)
            
            return speeches
            
        except Exception as e:
            st.error(f"Failed to load speeches from database: {str(e)}")
            traceback.print_exc()
            return []
    
    def load_topic_model(self, model_path=None):
        """Load BERTopic model from path or create a simple one for visualization"""
        try:
            if model_path and os.path.exists(model_path):
                model = BERTopic.load(model_path)
                print(f"✅ Loaded topic model from {model_path}")
                return model
            else:
                # Create a simple topic model with mock methods for visualization
                print("⚠️ No topic model found, creating a mock model")
                class MockTopicModel:
                    def __init__(self, speeches):
                        self.speeches = speeches
                        
                    def get_topic_info(self):
                        topics = {}
                        for s in self.speeches:
                            if s.topic is not None:
                                topics[s.topic] = s.topic_desc or f"Topic {s.topic}"
                        
                        df = pd.DataFrame({
                            'Topic': list(topics.keys()),
                            'Name': list(topics.values()),
                            'Count': [sum(1 for s in self.speeches if s.topic == t) for t in topics.keys()]
                        })
                        return df
                    
                    def get_topic(self, topic_id):
                        # Return mock topic words based on topic descriptions
                        speeches = [s for s in self.speeches if s.topic == topic_id]
                        if not speeches:
                            return []
                            
                        # Get words from topic description if available
                        if speeches[0].topic_desc:
                            words = speeches[0].topic_desc.split(", ")
                            return [(word, 0.9 - 0.1 * i) for i, word in enumerate(words)]
                        
                        return [("word1", 0.8), ("word2", 0.7), ("word3", 0.6)]
                    
                    def find_topics(self, keyword, top_n=5):
                        # Mock topic search - find topics with keyword in description
                        matching_topics = []
                        scores = []
                        
                        for s in self.speeches:
                            if s.topic is not None and s.topic_desc and keyword.lower() in s.topic_desc.lower():
                                if s.topic not in matching_topics:
                                    matching_topics.append(s.topic)
                                    scores.append(0.8)  # Mock score
                        
                        return matching_topics[:top_n], scores[:top_n]
                    
                    def save(self, path):
                        # Mock save function
                        os.makedirs(path, exist_ok=True)
                        with open(os.path.join(path, "mock_model.json"), "w") as f:
                            json.dump({"info": "Mock Topic Model"}, f)
                
                return MockTopicModel(speeches)
        except Exception as e:
            st.error(f"Failed to load topic model: {str(e)}")
            traceback.print_exc()
            return None

class Dashboard:
    def __init__(self, topic_model, embeddings_2d, speeches):
        self.tm = topic_model
        self.embeddings_2d = embeddings_2d
        self.speeches = speeches
        
    def run(self):
        #st.set_page_config(layout="wide", page_title="Speech Analysis Dashboard")
        
        #st.title("🧠 Parliamentary Speech Analysis Dashboard")
        
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Get unique values for filters
        factions = sorted(list(set([s.faction for s in self.speeches])))
        politicians = sorted(list(set([s.politician for s in self.speeches])))
        
        # Filter by faction
        selected_factions = st.sidebar.multiselect(
            "Select Factions", 
            options=factions,
            default=factions
        )
        
        # Filter by politician
        selected_politicians = st.sidebar.multiselect(
            "Select Politicians", 
            options=politicians,
            default=[]
        )
        
        # Date range filter
        try:
            dates = sorted(list(set([s.date for s in self.speeches])))
            min_date, max_date = min(dates), max(dates)
            date_range = st.sidebar.date_input(
                "Date Range",
                value=[datetime.strptime(min_date, "%Y-%m-%d").date(), 
                      datetime.strptime(max_date, "%Y-%m-%d").date()],
                min_value=datetime.strptime(min_date, "%Y-%m-%d").date(),
                max_value=datetime.strptime(max_date, "%Y-%m-%d").date(),
            )
        except:
            # If date parsing fails, skip date filtering
            st.sidebar.warning("Could not parse dates. Date filtering disabled.")
            date_range = None
        
        # Filter by topic
        topic_info = self.tm.get_topic_info()
        topic_options = [(row['Topic'], row['Name']) for _, row in topic_info.iterrows()]
        selected_topics = st.sidebar.multiselect(
            "Select Topics",
            options=[t[0] for t in topic_options],
            format_func=lambda x: f"Topic {x}: {dict(topic_options)[x][:50]}...",
            default=[]
        )
        
        # Apply filters
        filtered_indices = []
        for i, speech in enumerate(self.speeches):
            # Check if speech matches all filters
            faction_match = not selected_factions or speech.faction in selected_factions
            politician_match = not selected_politicians or speech.politician in selected_politicians
            topic_match = not selected_topics or speech.topic in selected_topics
            
            # Date filtering if enabled
            date_match = True
            if date_range and len(date_range) == 2:
                try:
                    speech_date = datetime.strptime(speech.date, "%Y-%m-%d").date()
                    date_match = date_range[0] <= speech_date <= date_range[1]
                except:
                    # If date parsing fails for a speech, include it anyway
                    pass
            
            if faction_match and politician_match and topic_match and date_match:
                filtered_indices.append(i)
        
        # Create filtered data
        filtered_speeches = [self.speeches[i] for i in filtered_indices]
        filtered_embeddings = self.embeddings_2d[filtered_indices] if len(filtered_indices) > 0 else np.array([])
        
        # Main content area - two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Topic distribution visualization
            st.subheader("Topic Distribution")
            
            if len(filtered_speeches) > 0 and filtered_embeddings.size > 0:
                # Prepare data for visualization
                speech_data = {
                    'x': filtered_embeddings[:, 0],
                    'y': filtered_embeddings[:, 1],
                    'faction': [s.faction for s in filtered_speeches],
                    'politician': [s.politician for s in filtered_speeches],
                    'topic': [s.topic for s in filtered_speeches],
                    'topic_desc': [s.topic_desc for s in filtered_speeches],
                    'date': [s.date for s in filtered_speeches],
                }
                
                df_viz = pd.DataFrame(speech_data)
                
                # Color by options
                color_by = st.radio("Color by:", ["Faction", "Topic"], horizontal=True)
                
                if color_by == "Faction":
                    color_column = 'faction'
                    color_discrete_map = {faction: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                                        for i, faction in enumerate(factions)}
                else:  # Topic
                    color_column = 'topic'
                    color_discrete_map = {topic: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                                        for i, topic in enumerate(sorted(list(set([s.topic for s in filtered_speeches if s.topic is not None]))))}
                
                # Create scatter plot
                fig = px.scatter(
                    df_viz, x='x', y='y', 
                    color=color_column,
                    color_discrete_map=color_discrete_map,
                    hover_data=['politician', 'topic_desc', 'date'],
                    title=f"2D Projection of Speeches ({len(filtered_speeches)} speeches shown)"
                )
                
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                fig.update_layout(
                    height=600,
                    xaxis_title="UMAP Dimension 1",
                    yaxis_title="UMAP Dimension 2",
                    legend_title=color_by
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No speeches match the selected filters or no 2D embeddings available.")
                
        with col2:
            # Topic explorer
            st.subheader("Topic Explorer")
            
            all_topics = sorted(list(set([s.topic for s in self.speeches if s.topic is not None])))
            if all_topics:
                topic_id = st.selectbox("Select Topic", all_topics)
                
                if topic_id is not None:
                    # Show topic details
                    topic_words = self.tm.get_topic(topic_id)
                    
                    st.write("**Top Words:**")
                    word_data = {
                        "Word": [w[0] for w in topic_words[:10]],
                        "Score": [w[1] for w in topic_words[:10]]
                    }
                    
                    word_df = pd.DataFrame(word_data)
                    fig = px.bar(word_df, x="Score", y="Word", orientation='h',
                                title=f"Topic {topic_id} - Top Words",
                                labels={"Score": "Relevance", "Word": ""},
                                height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Count speeches per faction in this topic
                    topic_speeches = [s for s in self.speeches if s.topic == topic_id]
                    faction_counts = {}
                    for s in topic_speeches:
                        faction_counts[s.faction] = faction_counts.get(s.faction, 0) + 1
                    
                    st.write("**Faction Distribution:**")
                    faction_df = pd.DataFrame({
                        "Faction": list(faction_counts.keys()),
                        "Count": list(faction_counts.values())
                    })
                    
                    fig = px.pie(faction_df, values="Count", names="Faction",
                                title=f"Topic {topic_id} - Faction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show example speeches for this topic
                    st.write("**Example Speeches:**")
                    max_examples = min(3, len(topic_speeches))
                    for i in range(max_examples):
                        with st.expander(f"{topic_speeches[i].politician} ({topic_speeches[i].faction}) - {topic_speeches[i].date}"):
                            st.write(topic_speeches[i].content[:500] + "..." if len(topic_speeches[i].content) > 500 else topic_speeches[i].content)
            else:
                st.warning("No topics found in the data. Run topic modeling first.")
            
            # Keyword search
            st.subheader("Topic Search")
            keyword = st.text_input("Search topics by keyword")
            if keyword:
                similar_topics = self.tm.find_topics(keyword, top_n=5)
                if similar_topics:
                    st.write("**Topics related to your search:**")
                    for topic, score in zip(similar_topics[0], similar_topics[1]):
                        topic_words = self.tm.get_topic(topic)
                        topic_desc = ", ".join([w[0] for w in topic_words[:5]]) if topic_words else "No description"
                        st.write(f"**Topic {topic}** (Score: {score:.2f}): {topic_desc}")
        
        # Bottom section - statistics and insights
        st.subheader("Speech Statistics")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            # Speeches per faction
            faction_counts = {}
            for s in filtered_speeches:
                faction_counts[s.faction] = faction_counts.get(s.faction, 0) + 1
            
            faction_df = pd.DataFrame({
                "Faction": list(faction_counts.keys()),
                "Count": list(faction_counts.values())
            }).sort_values("Count", ascending=False)
            
            fig = px.bar(faction_df, y="Faction", x="Count", orientation='h',
                         title="Speeches per Faction",
                         labels={"Count": "Number of Speeches", "Faction": ""},
                         height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_stats2:
            # Top politicians
            politician_counts = {}
            for s in filtered_speeches:
                politician_counts[s.politician] = politician_counts.get(s.politician, 0) + 1
            
            politician_df = pd.DataFrame({
                "Politician": list(politician_counts.keys()),
                "Count": list(politician_counts.values())
            }).sort_values("Count", ascending=False).head(10)
            
            fig = px.bar(politician_df, y="Politician", x="Count", orientation='h',
                         title="Top 10 Politicians by Number of Speeches",
                         labels={"Count": "Number of Speeches", "Politician": ""},
                         height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_stats3:
            # Topic distribution
            topic_counts = {}
            for s in filtered_speeches:
                if s.topic is not None and s.topic_desc:
                    topic_name = f"Topic {s.topic}: {s.topic_desc[:20]}..."
                    topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
            
            if topic_counts:
                topic_df = pd.DataFrame({
                    "Topic": list(topic_counts.keys()),
                    "Count": list(topic_counts.values())
                }).sort_values("Count", ascending=False).head(10)
                
                fig = px.bar(topic_df, y="Topic", x="Count", orientation='h',
                            title="Top 10 Topics",
                            labels={"Count": "Number of Speeches", "Topic": ""},
                            height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No topic data available for visualization.")
        
        # Export options
        st.subheader("Export Data")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("Export Filtered Speeches to CSV"):
                # Export current filtered speeches to CSV
                export_df = pd.DataFrame([{
                    "content": s.content,
                    "politician": s.politician,
                    "faction": s.faction,
                    "date": s.date,
                    "topic": s.topic,
                    "topic_desc": s.topic_desc,
                    "cluster": s.cluster
                } for s in filtered_speeches])
                
                if len(export_df) > 0:
                    # Create timestamp for filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = f"exported_speeches_{timestamp}.csv"
                    export_df.to_csv(csv_path, index=False)
                    st.success(f"✅ Exported {len(export_df)} speeches to {csv_path}")
                else:
                    st.warning("No speeches to export.")
        
        with col_export2:
            if st.button("Export Topic Model"):
                # Export the BERTopic model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"topic_model_{timestamp}"
                os.makedirs(model_path, exist_ok=True)
                self.tm.save(model_path)
                st.success(f"✅ Exported topic model to {model_path}")


def load_2d_embeddings(speeches, force_recalculate=False):
    """Generate or load 2D embeddings for visualization"""
    try:
        # Check if we have a cache file for 2D embeddings
        cache_file = "embeddings_2d_cache.npy"
        
        if os.path.exists(cache_file) and not force_recalculate:
            # Load from cache
            embeddings_2d = np.load(cache_file)
            print(f"✅ Loaded 2D embeddings from cache ({len(embeddings_2d)} points)")
            
            # Verify dimensions match
            if len(embeddings_2d) == len(speeches):
                print("Found caches 2d representations! Returning embeddings_2d")
                return embeddings_2d
            else:
                print(f"⚠️ Cache size mismatch: {len(embeddings_2d)} vs {len(speeches)} speeches")
                # Fall through to recalculation
        
        # No cache or force recalculate
        print("🔄 Generating 2D embeddings with UMAP...")
        
        # Extract embeddings from speech objects
        embeddings = np.array([s.embedding for s in speeches if s.embedding is not None])
        
        if len(embeddings) == 0:
            print("❌ No embeddings found in speeches!")
            # Return empty array with correct shape for compatibility
            return np.zeros((len(speeches), 2))
        
        from umap import UMAP
        
        # Generate 2D projection
        reducer = UMAP(n_components=2, min_dist=0.1, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Save to cache
        np.save(cache_file, embeddings_2d)
        print(f"✅ Generated and saved 2D embeddings ({len(embeddings_2d)} points)")
        
        return embeddings_2d
    except Exception as e:
        print(f"❌ Error generating 2D embeddings: {str(e)}")
        traceback.print_exc()
        # Return empty array with correct shape for compatibility
        return np.zeros((len(speeches), 2))


# Main entry point
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Speech Analysis Dashboard")
    
    # Database configuration
    DB_CONFIG = {
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432',
        'database': 'cluster',
        'document_table': 'speeches'
    }
    
    # Streamlit app header
    st.title("🧠 Parliamentary Speech Analysis Dashboard")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Option to use mock data if database is not accessible
    use_mock_data = st.sidebar.checkbox("Use mock data (if database is unavailable)", False)
    force_recalculate = st.sidebar.checkbox("Force recalculate 2D embeddings", False)
    
    # Option to load a custom topic model
    topic_model_path = st.sidebar.text_input("Topic model path (optional)", "")
    
    # Load speeches from database or use mock data
    try:
        if use_mock_data:
            # Create mock speeches if database is unavailable
            print("⚠️ Using mock data")
            speeches = []
            factions = ["SPD", "CDU/CSU", "GRÜNE", "FDP", "AfD", "LINKE"]
            politicians = ["Politician A", "Politician B", "Politician C"]
            topics = [0, 1, 2, 3, -1]  # -1 is "no topic"
            
            for i in range(50):
                faction = factions[i % len(factions)]
                politician = politicians[i % len(politicians)]
                topic = topics[i % len(topics)]
                
                speech = Speech(
                    content=f"This is a mock speech content for speech {i}. " * 5,
                    politician=politician,
                    term=20,
                    position="MdB",
                    date="2021-10-26",
                    faction=faction,
                    embedding=[0.1] * 384,  # Mock embedding
                    cluster=i % 3,
                    topic=topic,
                    topic_desc=f"Topic {topic}: mock topic" if topic >= 0 else "No topic"
                )
                speeches.append(speech)
                
            # Generate mock 2D embeddings
            import numpy as np
            embeddings_2d = np.random.rand(len(speeches), 2) * 10
            
            # Create mock topic model
            loader = DBLoader(DB_CONFIG)
            topic_model = loader.load_topic_model()
            
        else:
            # Load real data from database
            loader = DBLoader(DB_CONFIG)
            speeches = loader.load_speeches()
            
            if not speeches:
                st.error("❌ Failed to load speeches from database")
                st.stop()
                
            # Generate or load 2D embeddings
            embeddings_2d = load_2d_embeddings(speeches, force_recalculate)
            
            # Load or create topic model
            topic_model = loader.load_topic_model(topic_model_path if topic_model_path else None)
        
        # Launch dashboard
        dashboard = Dashboard(topic_model, embeddings_2d, speeches)
        dashboard.run()
        
    except Exception as e:
        st.error(f"❌ Error running dashboard: {str(e)}")
        traceback.print_exc()
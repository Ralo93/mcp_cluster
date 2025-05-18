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
    cache_file = "embeddings_2d_cache.npy"
    
    # Try to load from cache
    if os.path.exists(cache_file) and not force_recalculate:
        try:
            embeddings_2d = np.load(cache_file)
            if len(embeddings_2d) == len(speeches):
                print(f"✅ Loaded cached 2D embeddings")
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
            marker_size = st.slider("Marker Size", min_value=5, max_value=20, value=10, key="sidebar_marker_size")
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
        
        # Set color configuration based on selection
        if color_by == "Faction":
            color_column = 'faction'
            color_map = self.party_colors
            hover_data = ['politician', 'position', 'date', 'cluster']
        else:  # Cluster
            color_column = 'cluster'
            color_map = self.cluster_colors
            hover_data = ['politician', 'position', 'date', 'faction']


        with tab1:
            # Move color toggle from sidebar to top of visualization tab
            st.subheader("Visualization Controls")
            
            # Create a row for controls
            control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
            
            with control_col1:
                # Color scheme selection moved from sidebar - sync with session state
                tab1_color_by = st.radio(
                    "Color by:", 
                    ["Faction", "Cluster"], 
                    index=0 if st.session_state.color_by == "Faction" else 1,
                    horizontal=True,
                    key="tab1_color_by"
                )
                # Update session state when tab control changes
                st.session_state.color_by = tab1_color_by
                # Update local color_by variable
                color_by = tab1_color_by
                
                # Update color configuration based on new selection
                if color_by == "Faction":
                    color_column = 'faction'
                    color_map = self.party_colors
                    hover_data = ['politician', 'position', 'date', 'cluster']
                else:  # Cluster
                    color_column = 'cluster'
                    color_map = self.cluster_colors
                    hover_data = ['politician', 'position', 'date', 'faction']
            
            with control_col2:
                # Marker size control
                marker_size = st.slider("Marker Size", min_value=5, max_value=20, value=10, key="tab1_marker_size")
            
            with control_col3:
                # Show cluster labels
                show_cluster_labels = st.checkbox("Show cluster labels", value=True, key="tab1_show_labels")
            
            # Create two columns for visualization and speech details
            viz_col, speech_col = st.columns([2, 1])
            
            with viz_col:
                # Create interactive scatter plot with plotly
                fig = px.scatter(
                    df, x='x', y='y',
                    color=color_column,
                    color_discrete_map=color_map,
                    hover_data=hover_data,
                    custom_data=['speech_index', 'content_preview', 'topic', 'topic_desc', 'cluster_desc'],
                    height=700,
                    title="Speech Embedding Visualization"
                )
                
                # Enhanced hover template with topic and cluster descriptions
                hover_template = (
                    "<b>Speech %{customdata[0]}</b><br><br>" +
                    "Politician: %{customdata[1]}<br>" +
                    "Position: %{customdata[2]}<br>" +
                    "Date: %{customdata[3]}<br>"
                )
                
                # Add topic and cluster information to hover template based on color
                if color_by == "Faction":
                    hover_template += (
                        "Cluster: %{customdata[4]}<br>" +
                        "Cluster Description: %{customdata[6]}<br>" +
                        "Topic: %{customdata[5]}<br>" +
                        "Topic Description: %{customdata[7]}<br>" +
                        "Preview: %{customdata[8]}"
                    )
                else:  # Cluster
                    hover_template += (
                        "Faction: %{customdata[4]}<br>" +
                        "Cluster Description: %{customdata[6]}<br>" +
                        "Topic: %{customdata[5]}<br>" +
                        "Topic Description: %{customdata[7]}<br>" +
                        "Preview: %{customdata[8]}"
                    )
                
                # Configure marker appearance
                fig.update_traces(
                    marker=dict(size=marker_size, opacity=marker_opacity),
                    hovertemplate=hover_template
                )
                
                # Add cluster centroids if showing by cluster
                if color_by == "Cluster" and show_cluster_labels:
                    # Calculate cluster centroids
                    cluster_centroids = {}
                    for cluster_id in unique_clusters:
                        mask = df['cluster'] == cluster_id
                        if mask.any():
                            x_centroid = df.loc[mask, 'x'].mean()
                            y_centroid = df.loc[mask, 'y'].mean()
                            
                            # Get the most common cluster description for this cluster
                            cluster_descs = df.loc[mask, 'cluster_desc'].dropna()
                            most_common_desc = cluster_descs.mode()[0] if not cluster_descs.empty else f"Cluster {cluster_id}"
                            
                            cluster_centroids[cluster_id] = (x_centroid, y_centroid, most_common_desc)
                    
                    # Add text annotations for cluster labels with descriptions
                    for cluster_id, (x, y, desc) in cluster_centroids.items():
                        # Truncate description if too long
                        short_desc = f"{desc[:20]}..." if len(desc) > 20 else desc
                        
                        fig.add_annotation(
                            x=x, y=y,
                            text=f"Cluster {cluster_id}: {short_desc}",
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="#636363",
                            font=dict(size=12, color="#ffffff"),
                            bgcolor=self.cluster_colors.get(cluster_id, "#000000"),
                            bordercolor="#c7c7c7",
                            borderwidth=2,
                            borderpad=4,
                            opacity=0.9
                        )
                
                # Update layout
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="", 
                    yaxis_title="",
                    legend_title=color_by,
                    hovermode="closest"
                )
                
                # Use Streamlit's native chart click
                selected_points = st.plotly_chart(fig, use_container_width=True, key="scatter")
                
                # Store selected speech in session state
                if 'selected_speech_idx' not in st.session_state:
                    st.session_state.selected_speech_idx = None
                    
                # Handle click events - REMOVED the instruction message
                chart_data = st.session_state.get("scatter_clicked_data", None)
                if chart_data is not None and "points" in chart_data and len(chart_data["points"]) > 0:
                    point = chart_data["points"][0]
                    if "customdata" in point and len(point["customdata"]) > 0:
                        speech_idx = int(point["customdata"][0])
                        st.session_state.selected_speech_idx = speech_idx
                        
                # Add cluster and topic legend if coloring by cluster
                if color_by == "Cluster" and show_cluster_labels:
                    with st.expander("Cluster & Topic Descriptions", expanded=False):
                        # Create a concise legend for clusters and topics
                        legend_data = []
                        for cluster_id in sorted(unique_clusters):
                            mask = df['cluster'] == cluster_id
                            if mask.any():
                                # Get most common cluster description
                                cluster_descs = df.loc[mask, 'cluster_desc'].dropna()
                                cluster_desc = cluster_descs.mode()[0] if not cluster_descs.empty else ""
                                
                                # Get most common topics and their descriptions
                                topics = df.loc[mask, 'topic'].dropna()
                                topic_counts = topics.value_counts().head(3)
                                
                                # Collect top 3 topics for this cluster
                                top_topics = []
                                for topic_id, count in topic_counts.items():
                                    topic_mask = (df['cluster'] == cluster_id) & (df['topic'] == topic_id)
                                    if topic_mask.any():
                                        topic_descs = df.loc[topic_mask, 'topic_desc'].dropna()
                                        topic_desc = topic_descs.mode()[0] if not topic_descs.empty else ""
                                        top_topics.append((topic_id, topic_desc, count))
                                
                                legend_data.append({
                                    "Cluster": cluster_id,
                                    "Color": self.cluster_colors.get(cluster_id, "#000000"),
                                    "Description": cluster_desc,
                                    "Count": mask.sum(),
                                    "Top Topics": top_topics
                                })
                        
                        # Display legend
                        for item in legend_data:
                            color_box = f"<div style='background-color: {item['Color']}; width: 20px; height: 20px; display: inline-block; margin-right: 8px;'></div>"
                            st.markdown(f"{color_box} <b>Cluster {item['Cluster']}</b> ({item['Count']} speeches): {item['Description']}", unsafe_allow_html=True)
                            
                            if item['Top Topics']:
                                st.markdown("Top Topics:")
                                for topic_id, desc, count in item['Top Topics']:
                                    st.markdown(f"&nbsp;&nbsp;• Topic {topic_id} ({count} speeches): {desc}")
                                st.markdown("---")

        # Cluster Analysis tab
        with tab2:
            # Get unique clusters and count speeches
            clusters = {}
            for s in self.speeches:
                if s.cluster is not None:
                    cluster_id = s.cluster
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            "count": 0,
                            "topics": {},
                            "factions": {}
                        }
                    clusters[cluster_id]["count"] += 1
                    
                    # Record topics within this cluster
                    if s.topic is not None:
                        topic_id = s.topic
                        topic_text = s.topic_desc or f"Topic {topic_id}"
                        if topic_id not in clusters[cluster_id]["topics"]:
                            clusters[cluster_id]["topics"][topic_id] = {
                                "desc": topic_text,
                                "count": 0
                            }
                        clusters[cluster_id]["topics"][topic_id]["count"] += 1
                    
                    # Record faction distribution within cluster
                    faction = s.faction
                    if faction not in clusters[cluster_id]["factions"]:
                        clusters[cluster_id]["factions"][faction] = 0
                    clusters[cluster_id]["factions"][faction] += 1
            
            # Create columns for cluster stats and visualization
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Cluster Overview")
                
                # Create a summary table
                cluster_summary = []
                for cluster_id in sorted(clusters.keys()):
                    cluster_data = clusters[cluster_id]
                    cluster_summary.append({
                        "Cluster ID": cluster_id,
                        "Speeches": cluster_data["count"],
                        "Top Topics": ", ".join([f"{topic_id}" for topic_id, _ in 
                                            sorted(cluster_data["topics"].items(), 
                                                key=lambda x: x[1]["count"], 
                                                reverse=True)[:3]]),
                        "Dominant Faction": max(cluster_data["factions"].items(), 
                                              key=lambda x: x[1])[0] if cluster_data["factions"] else "N/A"
                    })
                
                if cluster_summary:
                    cluster_df = pd.DataFrame(cluster_summary)
                    st.dataframe(cluster_df, use_container_width=True)
                else:
                    st.warning("No cluster data available")
            
            with col2:
                st.subheader("Cluster Distribution")
                
                # Create data for stacked bar chart
                if clusters:
                    # Prepare data for stacked bar chart showing faction distribution within clusters
                    cluster_faction_data = []
                    
                    # Get all unique factions across clusters
                    all_factions = set()
                    for cluster_data in clusters.values():
                        all_factions.update(cluster_data["factions"].keys())
                    
                    # Create data entries for each cluster and faction
                    for cluster_id in sorted(clusters.keys()):
                        cluster_data = clusters[cluster_id]
                        
                        # Add an entry for each faction
                        for faction in all_factions:
                            count = cluster_data["factions"].get(faction, 0)
                            if count > 0:  # Only include non-zero values
                                cluster_faction_data.append({
                                    "Cluster": f"Cluster {cluster_id}",
                                    "Faction": faction,
                                    "Count": count
                                })
                    
                    # Convert to DataFrame for plotting
                    if cluster_faction_data:
                        df_plot = pd.DataFrame(cluster_faction_data)
                        
                        # Create stacked bar chart
                        fig = px.bar(
                            df_plot,
                            x="Cluster",
                            y="Count",
                            color="Faction",
                            color_discrete_map=self.party_colors,
                            title="Faction Distribution per Cluster",
                            labels={"Count": "Number of Speeches", "Cluster": "Cluster ID"}
                        )
                        
                        # Improve layout
                        fig.update_layout(
                            legend_title="Faction",
                            xaxis_title="Cluster ID",
                            yaxis_title="Number of Speeches",
                            hovermode="closest"
                        )
                        
                        # Add total count annotations
                        for cluster_id in sorted(clusters.keys()):
                            total = clusters[cluster_id]["count"]
                            if total > 0:
                                fig.add_annotation(
                                    x=f"Cluster {cluster_id}",
                                    y=total,
                                    text=f"{total}",
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color="black", size=12)
                                )
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a toggle to show cluster composition as a table
                        if st.checkbox("Show cluster composition as table"):
                            # Pivot the data for a better table view
                            pivot_df = df_plot.pivot(index="Cluster", columns="Faction", values="Count").fillna(0).astype(int)
                            pivot_df["Total"] = pivot_df.sum(axis=1)
                            st.dataframe(pivot_df, use_container_width=True)
                    else:
                        st.warning("No faction distribution data available")
                else:
                    st.warning("No cluster data available")
            
            # Select a cluster to analyze
            if clusters:
                selected_cluster = st.selectbox(
                    "Select a cluster to analyze:",
                    sorted(clusters.keys())
                )
                
                if selected_cluster is not None:
                    cluster_data = clusters[selected_cluster]
                    
                    # Create two columns for topics and faction distribution
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader(f"Topics in Cluster {selected_cluster}")
                        
                        # Sort topics by count
                        sorted_topics = sorted(
                            cluster_data["topics"].items(),
                            key=lambda x: x[1]["count"],
                            reverse=True
                        )
                        
                        # Display topic information
                        for topic_id, topic_info in sorted_topics:
                            st.markdown(f"### Topic {topic_id} ({topic_info['count']} speeches)")
                            st.markdown(f"**Description:** {topic_info['desc']}")
                            
                            # Get example speeches for this topic
                            example_speeches = [
                                s for s in self.speeches 
                                if s.cluster == selected_cluster and s.topic == topic_id
                            ][:3]  # Limit to 3 examples
                            
                            if example_speeches:
                                st.markdown("**Example Speeches:**")
                                for i, speech in enumerate(example_speeches):
                                    st.markdown(f"**Speech {i+1} by {speech.politician} ({speech.date}):**")
                                    st.markdown(f"*Faction: {speech.faction}, Position: {speech.position}*")
                                    st.text_area(f"Content {i+1}", value=speech.content[:300] + "..." if len(speech.content) > 300 else speech.content, height=100)
                    
                    with col2:
                        st.subheader(f"Faction Distribution in Cluster {selected_cluster}")
                        
                        # Create a pie chart of faction distribution
                        faction_data = cluster_data["factions"]
                        
                        if faction_data:
                            fig = px.pie(
                                values=list(faction_data.values()),
                                names=list(faction_data.keys()),
                                color=list(faction_data.keys()),
                                color_discrete_map=self.party_colors
                            )
                            
                            fig.update_layout(
                                title="Faction Distribution",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No faction data available")
            else:
                st.warning("No clusters available for analysis")
        
        # Data Table tab
        with tab3:
            st.subheader("Speech Data Table")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter by faction
                factions = ["All"] + sorted(set(s.faction for s in self.speeches if s.faction))
                selected_faction = st.selectbox("Filter by Faction:", factions)
            
            with col2:
                # Filter by cluster
                clusters = ["All"] + sorted(set(str(s.cluster) for s in self.speeches if s.cluster is not None))
                selected_cluster = st.selectbox("Filter by Cluster:", clusters)
            
            with col3:
                # Filter by politician
                politicians = ["All"] + sorted(set(s.politician for s in self.speeches if s.politician))
                selected_politician = st.selectbox("Filter by Politician:", politicians)
            
            # Apply filters to dataframe
            filtered_df = df.copy()
            
            if selected_faction != "All":
                filtered_df = filtered_df[filtered_df['faction'] == selected_faction]
            
            if selected_cluster != "All":
                filtered_df = filtered_df[filtered_df['cluster'].astype(str) == selected_cluster]
            
            if selected_politician != "All":
                filtered_df = filtered_df[filtered_df['politician'] == selected_politician]
            
            # Display filtered dataframe
            if not filtered_df.empty:
                # Select columns to display
                display_columns = ['politician', 'faction', 'position', 'date', 'cluster', 'topic_desc', 'content_preview']
                display_df = filtered_df[display_columns].copy()
                
                # Rename columns for better display
                display_df.columns = ['Politician', 'Faction', 'Position', 'Date', 'Cluster', 'Topic', 'Content Preview']
                
                st.dataframe(display_df, use_container_width=True)
                
                st.text(f"Showing {len(display_df)} speeches out of {len(df)} total")
            else:
                st.warning("No speeches match the selected filters")


        # Speech Content by Cluster tab
        with tab4:
            st.subheader("Speech Content by Cluster")
            
            # Get unique clusters for selection

            #TODO CHANGED
            all_clusters = sorted(set(s.topic_desc for s in self.speeches if s.topic_desc is not None)) 
            
            if all_clusters:
                # Cluster selection
                selected_cluster_id = st.selectbox(
                    "Select a cluster to inspect speeches:",
                    all_clusters,
                    key="content_cluster_selector"
                )
                
                # Get all speeches in the selected cluster
                #TODO CHANGED
                cluster_speeches = [s for s in self.speeches if s.topic_desc == selected_cluster_id]
                
                if cluster_speeches:
                    # Display cluster information
                    st.markdown(f"### Cluster {selected_cluster_id} - {len(cluster_speeches)} speeches")
                    
                    # Show color legend for this cluster
                    if selected_cluster_id in self.cluster_colors:
                        st.markdown(
                            f"<div style='background-color: {self.cluster_colors[selected_cluster_id]}; "
                            f"width: 20px; height: 20px; display: inline-block; margin-right: 8px;'></div> "
                            f"<span>Cluster {selected_cluster_id} color</span>", 
                            unsafe_allow_html=True
                        )
                    
                    # Additional filters
                    col1, col2 = st.columns(2)
                    with col1:
                        # Filter by faction within cluster
                        cluster_factions = ["All"] + sorted(set(s.faction for s in cluster_speeches if s.faction))
                        selected_faction = st.selectbox(
                            "Filter by faction:", 
                            cluster_factions,
                            key="content_faction_filter"
                        )
                    
                    with col2:
                        # Filter by politician within cluster
                        cluster_politicians = ["All"] + sorted(set(s.politician for s in cluster_speeches if s.politician))
                        selected_politician = st.selectbox(
                            "Filter by politician:", 
                            cluster_politicians,
                            key="content_politician_filter"
                        )
                    
                    # Apply filters
                    filtered_speeches = cluster_speeches
                    if selected_faction != "All":
                        filtered_speeches = [s for s in filtered_speeches if s.faction == selected_faction]
                    if selected_politician != "All":
                        filtered_speeches = [s for s in filtered_speeches if s.politician == selected_politician]
                    
                    # Display count after filtering
                    st.markdown(f"Showing {len(filtered_speeches)} speeches from Cluster {selected_cluster_id}")
                    
                    # Option to limit speech length for readability
                    max_chars = st.slider("Maximum characters to display", 
                                        min_value=100, 
                                        max_value=2000, 
                                        value=500, 
                                        step=100)
                    
                    # Speech viewer
                    for i, speech in enumerate(filtered_speeches):
                        with st.expander(
                            f"Speech by {speech.politician} ({speech.faction}) - {speech.date}",
                            expanded=(i == 0)  # Expand first speech by default
                        ):
                            # Metadata
                            st.markdown(f"**Position:** {speech.position}")
                            if speech.topic is not None:
                                st.markdown(f"**Topic:** {speech.topic} - {speech.topic_desc}")
                            
                            # Speech content with limitation
                            display_content = speech.content
                            if len(display_content) > max_chars:
                                display_content = display_content[:max_chars] + "..."
                                st.text_area("Speech Content (truncated)", value=display_content, height=250)
                                if st.button(f"Show Full Speech {i+1}", key=f"show_full_{i}"):
                                    st.text_area("Full Speech Content", value=speech.content, height=400)
                            else:
                                st.text_area("Speech Content", value=display_content, height=250)
                else:
                    st.warning(f"No speeches found in Cluster {selected_cluster_id}")
            else:
                st.warning("No clusters available")

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
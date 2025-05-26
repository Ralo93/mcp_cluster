"""
Modern Knowledge Graph Builder using LlamaIndex v0.10+ APIs
Supports multiple LLM providers and enhanced visualization
"""

import os
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex, Settings
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter

# Modern LLM imports - choose your provider
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
#from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@dataclass
class KGConfig:
    """Configuration for Knowledge Graph construction"""
    llm_provider: str = "openai"  # openai, anthropic, ollama
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_triplets_per_chunk: int = 10
    chunk_size: int = 1024
    chunk_overlap: int = 200
    include_embeddings: bool = False
    max_chars_per_doc: int = 2000

class ModernKnowledgeGraphBuilder:
    """Modern Knowledge Graph Builder with enhanced capabilities"""
    
    def __init__(self, config: KGConfig):
        self.config = config
        self._setup_llm_and_embeddings()
        self._setup_storage_context()
    
    def _setup_llm_and_embeddings(self):
        """Setup LLM and embedding models based on configuration"""
        if self.config.llm_provider == "openai":
            self.llm = OpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=os.getenv("OPENAI_API_KEY")  # ✅ inject key directly
            )
            if self.config.include_embeddings:
                self.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        
        elif self.config.llm_provider == "anthropic":
            self.llm = Anthropic(
                model=self.config.model_name or "claude-3-sonnet-20240229",
                temperature=self.config.temperature
            )
       #     if self.config.include_embeddings:
       #         self.embed_model = HuggingFaceEmbedding(
       #             model_name="BAAI/bge-large-en-v1.5"
       #         )
        
        #elif self.config.llm_provider == "ollama":
        #    self.llm = Ollama(
        #        model=self.config.model_name or "llama3.1:8b",
        #        temperature=self.config.temperature
        #    )
        #    if self.config.include_embeddings:
        #        self.embed_model = HuggingFaceEmbedding(
        #            model_name="BAAI/bge-large-en-v1.5"
        #        )
        
        # Configure global settings
        Settings.llm = self.llm
        if self.config.include_embeddings:
            Settings.embed_model = self.embed_model
    
    def _setup_storage_context(self):
        """Setup storage context with graph store"""
        self.graph_store = SimpleGraphStore()
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )
    
    def df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """Convert DataFrame rows to LlamaIndex documents with enhanced metadata"""
        documents = []
        
        for idx, row in df.iterrows():
            # Enhanced metadata structure
            metadata = {
                "politician": row.get('politician', ''),
                "faction": row.get('faction', ''),
                "term": row.get('term', ''),
                "date": str(row.get('date', '')),
                "topic": row.get('topic_desc', ''),
                "cluster": row.get('cluster_desc', ''),
                "doc_id": f"speech_{idx}",
                "source": "parliamentary_speeches"
            }
            
            # Create readable text representation
            readable_metadata = (
                f"Politician: {metadata['politician']}\n"
                f"Faction: {metadata['faction']}\n"
                f"Term: {metadata['term']}\n"
                f"Date: {metadata['date']}\n"
                f"Topic: {metadata['topic']}\n"
                f"Cluster: {metadata['cluster']}"
            )
            
            content = (row.get("content") or "")[:self.config.max_chars_per_doc]
            full_text = readable_metadata + "\n\n" + content
            
            doc = Document(
                text=full_text,
                metadata=metadata,
                doc_id=metadata["doc_id"]
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents from DataFrame")
        return documents
    
    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """Build knowledge graph with modern LlamaIndex APIs"""
        logger.info("Building knowledge graph index...")
        
        # Setup node parser for better chunking
        node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Setup keyword extractor for better entity recognition
        keyword_extractor = KeywordExtractor(
            keywords=20,  # Extract top 20 keywords per chunk
            llm=self.llm
        )
        
        # Build the knowledge graph index
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            node_parser=node_parser,
            max_triplets_per_chunk=self.config.max_triplets_per_chunk,
            include_embeddings=self.config.include_embeddings,
            show_progress=True
        )
        
        logger.info("Knowledge graph index built successfully")
        return kg_index
    
    def get_graph_statistics(self, kg_index: KnowledgeGraphIndex) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        try:
            triplets = kg_index.get_triplets()
            
            # Create NetworkX graph for analysis
            G = nx.DiGraph()
            entities = set()
            relations = set()
            
            for subj, pred, obj in triplets:
                G.add_edge(subj, obj, relation=pred)
                entities.add(subj)
                entities.add(obj)
                relations.add(pred)
            
            stats = {
                "total_triplets": len(triplets),
                "unique_entities": len(entities),
                "unique_relations": len(relations),
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_weakly_connected(G),
                "number_of_components": nx.number_weakly_connected_components(G)
            }
            
            # Add centrality measures for top entities
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                stats["top_central_entities"] = top_entities
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating graph statistics: {e}")
            return {"error": str(e)}
    
    def create_interactive_visualization(self, kg_index: KnowledgeGraphIndex) -> go.Figure:
        """Create an enhanced interactive visualization of the knowledge graph"""
        try:
            triplets = kg_index.get_triplets()
            
            if not triplets:
                logger.warning("No triplets found in knowledge graph")
                return go.Figure()
            
            # Build NetworkX graph
            G = nx.DiGraph()
            edge_labels = {}
            
            for subj, pred, obj in triplets:
                G.add_edge(subj, obj, relation=pred, weight=1)
                edge_labels[(subj, obj)] = pred
            
            # Use better layout algorithm
            if G.number_of_nodes() > 100:
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Calculate node sizes based on degree centrality
            centrality = nx.degree_centrality(G)
            node_sizes = [centrality.get(node, 0.1) * 50 + 10 for node in G.nodes()]
            
            # Create edge traces
            edge_x, edge_y = [], []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                relation = G.edges[edge].get('relation', '')
                edge_info.append(f"{edge[0]} → {edge[1]}: {relation}")
            
            # Create node traces
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            node_info = [f"Entity: {node}<br>Connections: {G.degree(node)}<br>Centrality: {centrality.get(node, 0):.3f}" 
                        for node in G.nodes()]
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=("Knowledge Graph Visualization",)
            )
            
            # Add edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                mode='lines',
                name='Relations'
            )
            
            # Add node trace with enhanced styling
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_info,
                marker=dict(
                    size=node_sizes,
                    color=[centrality.get(node, 0) for node in G.nodes()],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Centrality"),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name='Entities'
            )
            
            fig.add_trace(edge_trace)
            fig.add_trace(node_trace)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Parliamentary Speeches Knowledge Graph<br><sub>Entities: {G.number_of_nodes()}, Relations: {G.number_of_edges()}</sub>',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=80),
                annotations=[
                    dict(
                        text="Node size represents centrality. Hover for details.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12, color="grey")
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return go.Figure().add_annotation(text=f"Visualization error: {e}")

def main():
    """Main execution function with enhanced error handling and logging"""

    from dotenv import load_dotenv
    load_dotenv()

    # Check for API keys
    
    try:
        # Configuration
        config = KGConfig(
            llm_provider="openai",  # Change to "anthropic" or "ollama" as needed
            model_name="gpt-3.5-turbo-1106",     # or "claude-3-sonnet-20240229" for Anthropic
            temperature=0.0,
            max_triplets_per_chunk=15,
            chunk_size=1024,
            include_embeddings=False
        )
        
        print("🔄 Loading speeches from database...")
        
        # Import your data loader (assuming it exists)
        from data_loader import DataLoader, build_db_url
        
        CSV_PATH = "data/leg20.csv"
        DB_CONFIG = {
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432',
            'database': 'cluster',
            'document_table': 'speeches'
        }
        
        # Load data
        loader = DataLoader(CSV_PATH, DB_CONFIG)
        speeches_df = loader.load_from_db(amount=50)
        
        if speeches_df is None or speeches_df.empty:
            raise ValueError("No speeches found in the database.")
        
        print(f"✅ Loaded {len(speeches_df)} speeches from database.")
        
        # Initialize knowledge graph builder
        kg_builder = ModernKnowledgeGraphBuilder(config)
        
        # Convert to documents
        documents = kg_builder.df_to_documents(speeches_df)
        
        # Build knowledge graph
        kg_index = kg_builder.build_knowledge_graph(documents)
        
        # Get statistics
        stats = kg_builder.get_graph_statistics(kg_index)
        
        print("\n📊 Knowledge Graph Statistics:")
        for key, value in stats.items():
            if key != "top_central_entities":
                print(f"  {key}: {value}")
        
        if "top_central_entities" in stats:
            print("\n🔝 Most Central Entities:")
            for entity, centrality in stats["top_central_entities"][:5]:
                print(f"  {entity}: {centrality:.3f}")
        
        # Create and show visualization
        print("\n🎨 Creating visualization...")
        fig = kg_builder.create_interactive_visualization(kg_index)
        fig.show()
        
        # Save the knowledge graph for later use
        kg_index.storage_context.persist(persist_dir="./kg_storage")
        print("\n💾 Knowledge graph saved to ./kg_storage")
        
        return kg_index, stats
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":


    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    print("🔐 Loaded key:", os.getenv("OPENAI_API_KEY") is not None)
    kg_index, stats = main()
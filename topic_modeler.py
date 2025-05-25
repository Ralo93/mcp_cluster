# Enhanced TopicModeler Class

import json
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

from bertopic.representation import MaximalMarginalRelevance

from sklearn.feature_extraction.text import CountVectorizer


# Gold standard by manual fine-tuning
CUSTOM_STOPWORDS = {

    "herr", "wirklich", "dem", "war", "deshalb", 
    "doch", "natürlich", "sondern", "deswegen", 
    "dazu", "dr", "deswegen", "dafür", "gerade",
    "uns", "an", "hat", "was", "ihnen", "im", 
    "man", "jetzt", "muss", "müssen", "brauchen", 
    "gesagt", "dank", "unsere", "dank", "sagen", 
    "dann", "dieser", "diese", "diesen", "als", 
    "weil", "gesagt", "hier", "heute", "damit", 
    "daher", "des", "ein", "eine", "auf", "dass", 
    "noch", "mal", "bitte", "vielleicht", "sie", 
    "wir", "nicht", "in", "zu", "auch", "es", 
    "der", "bei", "den" ,"haben", "aber", 
    "präsidentin", "diesem", "denn", "du"
}


# Create CountVectorizer with custom stopwords
vectorizer = CountVectorizer(stop_words=list(CUSTOM_STOPWORDS)) # Can also be used with min_df for automatic stop word detection using term frequencies

class TopicModeler:
    def __init__(self, embedding_model, umap_model, hdbscan_model):
        # Initialize KeyBERTInspired representation model

        self.representation_model = KeyBERTInspired()
        #self.representation_model = MaximalMarginalRelevance(diversity=0.6)

        # Initialize BERTopic with the representation model
        self.model = BERTopic(
            language="german",
            #vectorizer_model=vectorizer, # not working - use later in update_topics which actually works
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=self.representation_model,
            verbose=True
        )
        
        # Track original topics before updates
        self.original_topics = None

    def fit(self, texts, embeddings):
        """Fit the model and update topics with better representations"""
        # Fit the basic model
        self.model.fit(texts, embeddings)
        
        # Store original topics for comparison
        self.original_topics = self.model.get_topics()
        
        # Update with KeyBERTInspired representation
        self.update_topics(texts)
        
        return self.model

    def update_topics(self, texts):
        """Update topic representations with the KeyBERTInspired model"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        try:
            # Update topics with KeyBERTInspired representation
            self.model.update_topics(texts, representation_model=self.representation_model,
            vectorizer_model=vectorizer  #  inject the custom stopword-aware vectorizer #TODO This actually works!
        )
            print("✅ Successfully updated topics with KeyBERTInspired representation")
        except Exception as e:
            print(f"⚠️ Failed to update topics: {str(e)}")
    
    def transform(self, texts, embeddings=None):
        """Transform texts to topics, handling cases where prediction data might be missing"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        try:
            # Try standard transform
            topics, probs = self.model.transform(texts, embeddings)
            return topics, probs
        except AttributeError as e:
            # If prediction_data is missing, we need to work around it
            if "No prediction data was generated" in str(e):
                print("⚠️ Warning: Prediction data was missing. Using fit_transform instead.")
                # Use fit_transform which will regenerate everything
                topics, probs = self.model.fit_transform(texts, embeddings)
                return topics, probs
            else:
                # Different error, re-raise
                raise

    def explore_topic(self, topic_id=0):
        """Get details about a specific topic"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.model.get_topic(topic_id)

    def search_topic(self, keyword):
        """Search for topics containing a specific keyword"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.model.find_topics(keyword)
    
    def get_topic_info(self):
        """Get comprehensive information about all topics"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.model.get_topic_info()
    
    def topic_differences(self):
        """Compare original and updated topics to see the differences"""
        if self.model is None or self.original_topics is None:
            raise ValueError("Model not fitted yet or original topics not stored. Call fit() first.")
        
        updated_topics = self.model.get_topics()
        
        differences = {}
        for topic_id in self.original_topics.keys():
            if topic_id in updated_topics:
                original_words = set([word for word, _ in self.original_topics[topic_id]])
                updated_words = set([word for word, _ in updated_topics[topic_id]])
                
                # Find words added and removed
                added = updated_words - original_words
                removed = original_words - updated_words
                
                differences[topic_id] = {
                    "original": self.original_topics[topic_id],
                    "updated": updated_topics[topic_id],
                    "added": added,
                    "removed": removed
                }
        
        return differences
    
    def save_topic_descriptions_to_db(self, db_connection, table_name="topic_descriptions"):
        """Save topic descriptions to database for dashboard use"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get topic information
        topic_info = self.model.get_topic_info()
        topic_data = []
        
        # Create records for each topic
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip the outlier topic
                # Get the full topic description (words and weights)
                # Custom list of words to exclude
                EXCLUDE_WORDS = {"herr", "wirklich", "ein", "eine", "auf", "dass", "noch", "mal", "bitte", "vielleicht"}

                # Get the full topic
                raw_topic_words = self.model.get_topic(topic_id)

                # Filter out excluded words
                topic_words = [(word, score) for word, score in raw_topic_words if word.lower() not in EXCLUDE_WORDS]

                
                # Create a concise description from top words
                description = ", ".join([word for word, _ in topic_words[:15]])
                
                # Create record
                topic_data.append({
                    "topic_id": int(topic_id),
                    "name": f"Topic {topic_id}",
                    "description": description,
                    "count": int(row['Count']),
                    "words_json": json.dumps(topic_words[:15])  # Store top 10 words as JSON
                })
        
        # Convert to DataFrame for easier DB insertion
        topics_df = pd.DataFrame(topic_data)
        
        try:
            # Save to database
            topics_df.to_sql(table_name, db_connection, if_exists='replace', index=False)
            print(f"✅ Successfully saved {len(topics_df)} topic descriptions to database table '{table_name}'")
            return True
        except Exception as e:
            print(f"❌ Failed to save topic descriptions to database: {str(e)}")
            return False
    
    def visualize_topics(self, width=950, height=600):
        """Generate interactive topic visualization"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.model.visualize_topics(width=width, height=height)
    
    def visualize_documents(self, docs, embeddings, width=950, height=600):
        """Generate interactive document visualization colored by topics"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.model.visualize_documents(docs, embeddings, width=width, height=height)


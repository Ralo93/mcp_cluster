from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import psycopg2
from transformers import AutoTokenizer

class Embedder:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", db_url=None): #"thenlper/gte-small", "TUM/GottBERT_base_last", 
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.db_url = db_url

    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
    
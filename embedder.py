from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer

class Embedder:
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-de", db_url=None, max_chars=2000): #"thenlper/gte-small", "TUM/GottBERT_base_last", jina-embeddings-v2-base-de, paraphrase-multilingual-MiniLM-L12-v2
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.db_url = db_url
        self.max_chars = max_chars  # safe limit
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, texts, batch_size):
        print(f"Encoding with {self.device}")
        # Truncate texts before encoding
        truncated = [text[:self.max_chars] for text in texts]
        return self.model.encode(truncated, show_progress_bar=True, batch_size=batch_size, device=self.device)
    



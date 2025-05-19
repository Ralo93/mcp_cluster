## Parliament Agent system


## Considerations for Embedding Models:

| Feature       | GTE-Small | LaBSE | M-E5-Large    | BGE-M3        | M-MiniLM | USE (Distil) |
| ------------- | --------- | ----- | ------------- | ------------- | -------- | ------------ |
| Vector Dim    | 384       | 768   | 1024          | 1024          | 384      | 512          |
| German        | ✅         | ✅     | ✅             | ✅             | ✅        | ✅            |
| Speed         | ⚡⚡⚡       | ⚡     | ⚡             | ⚡             | ⚡⚡⚡      | ⚡⚡           |
| Accuracy      | Medium    | High  | **Very High** | **Very High** | Medium   | High         |
| Ideal for RAG | ❌         | ✅     | ✅✅            | ✅✅            | ❌        | ✅            |

## Embedding Model:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

## Considerations:

EMBEDDING MODEL
LENGTH OF WORDS
MIN CLUSTER SIZE
UMAP DIMENSION
CUSTOM STOPWORDS
KeyBERTInspired() VS MaximalMarginalRelevance


from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("Cached.")

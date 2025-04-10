from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", module="transformers")


def get_sentence_embedding(sentence):
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('D:/AI/model/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings

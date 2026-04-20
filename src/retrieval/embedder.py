import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from src.config import MODEL_CONFIGS


def load_embedding_model(model_key):
    config = MODEL_CONFIGS[model_key]
    print(f"Loading {model_key}: {config['model_name']}...")
    model = SentenceTransformer(config['model_name'])
    print(f"  Loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_faiss_index(chunks, model, model_key, batch_size=64):
    config = MODEL_CONFIGS[model_key]
    texts = [config['passage_prefix'] + c['text'] for c in chunks]

    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embed_time = time.time() - start_time

    embeddings = np.array(embeddings, dtype='float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"  {len(chunks)} chunks -> {index.ntotal} vectors ({embeddings.shape[1]}d) "
          f"in {embed_time:.1f}s ({len(chunks)/embed_time:.0f} chunks/sec)")

    return index, embeddings


def query_index(query, index, chunks, model, model_key, top_k=5):
    config = MODEL_CONFIGS[model_key]
    prefixed_query = config['query_prefix'] + query

    query_embedding = model.encode(
        [prefixed_query], normalize_embeddings=True
    ).astype('float32')

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        results.append({
            'rank': rank + 1,
            'score': float(score),
            'meeting_date': chunk['meeting_date'],
            'doc_type': chunk['doc_type'],
            'chunk_index': chunk['chunk_index'],
            'text': chunk['text'],
            'text_preview': chunk['text'][:200],
        })

    return results

import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi


def build_bm25_index(chunks):
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built: {len(tokenized)} documents, "
          f"avg length {np.mean([len(t) for t in tokenized]):.0f} tokens")
    return bm25


def save_indices(all_faiss_indices, all_chunk_sets, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)

    for model_key in all_faiss_indices:
        for size in all_faiss_indices[model_key]:
            faiss.write_index(
                all_faiss_indices[model_key][size],
                f'{save_dir}/index_{model_key}_{size}.faiss'
            )

    with open(f'{save_dir}/chunk_sets.pkl', 'wb') as f:
        pickle.dump(all_chunk_sets, f)

    total = sum(len(v) for v in all_faiss_indices.values())
    print(f"Saved {total} indices to {save_dir}/")


def load_faiss_index(path):
    return faiss.read_index(path)


def load_chunk_sets(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

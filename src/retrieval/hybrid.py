import re
import calendar
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.config import MODEL_CONFIGS
from src.retrieval.embedder import query_index


def parse_date_from_query(query):
    query_lower = query.lower()

    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
    }

    for month_name, month_num in months.items():
        match = re.search(rf'{month_name}\s+(\d{{4}})', query_lower)
        if match:
            year = int(match.group(1))
            last_day = calendar.monthrange(year, month_num)[1]
            return f'{year}-{month_num:02d}-01', f'{year}-{month_num:02d}-{last_day:02d}'

    match = re.search(r'(early|first half of|beginning of)\s+(\d{4})', query_lower)
    if match:
        year = int(match.group(2))
        return f'{year}-01-01', f'{year}-06-30'

    match = re.search(r'(late|second half of|end of)\s+(\d{4})', query_lower)
    if match:
        year = int(match.group(2))
        return f'{year}-07-01', f'{year}-12-31'

    match = re.search(r'(mid|middle of)\s+(\d{4})', query_lower)
    if match:
        year = int(match.group(2))
        return f'{year}-04-01', f'{year}-09-30'

    match = re.search(r'q([1-4])\s+(\d{4})', query_lower)
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        last_day = calendar.monthrange(year, end_month)[1]
        return f'{year}-{start_month:02d}-01', f'{year}-{end_month:02d}-{last_day:02d}'

    match = re.search(r'(\d{4})\s*(?:to|\u2013|-)\s*(\d{4})', query_lower)
    if match:
        year1 = int(match.group(1))
        year2 = int(match.group(2))
        return f'{year1}-01-01', f'{year2}-12-31'

    match = re.search(r'(?:in|during|of)\s+(\d{4})', query_lower)
    if match:
        year = int(match.group(1))
        return f'{year}-01-01', f'{year}-12-31'

    return None, None


def bm25_search(query, bm25, chunks, top_k=20):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = chunks[idx]
        results.append({
            'rank': rank + 1,
            'score': float(scores[idx]),
            'meeting_date': chunk['meeting_date'],
            'doc_type': chunk['doc_type'],
            'chunk_index': chunk['chunk_index'],
            'text': chunk['text'],
            'text_preview': chunk['text'][:200],
            'source': 'bm25',
        })

    return results


def hybrid_retrieve(query, faiss_index, bm25_index, chunks,
                    embedding_model, model_key,
                    top_k=5, semantic_top_n=20, bm25_top_n=20,
                    rrf_k=60, use_date_filter=True):
    start_date, end_date = parse_date_from_query(query) if use_date_filter else (None, None)

    if start_date and end_date:
        date_filtered_indices = [
            i for i, c in enumerate(chunks)
            if start_date <= c['meeting_date'] <= end_date
        ]
        filtered_chunks = [chunks[i] for i in date_filtered_indices]

        if len(filtered_chunks) == 0:
            semantic_results = query_index(
                query, faiss_index, chunks, embedding_model, model_key,
                top_k=semantic_top_n
            )
            for r in semantic_results:
                r['source'] = 'semantic'
            bm25_results = bm25_search(query, bm25_index, chunks, top_k=bm25_top_n)
            semantic_filtered = semantic_results
            bm25_filtered = bm25_results
            filter_note = f"{start_date} to {end_date} (no chunks found, fell back to full corpus)"
        else:
            config = MODEL_CONFIGS[model_key]

            filtered_texts = [config['passage_prefix'] + c['text'] for c in filtered_chunks]
            filtered_embeddings = embedding_model.encode(
                filtered_texts, normalize_embeddings=True
            ).astype('float32')

            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings)

            query_embedding = embedding_model.encode(
                [config['query_prefix'] + query], normalize_embeddings=True
            ).astype('float32')

            sem_k = min(semantic_top_n, len(filtered_chunks))
            scores, indices = temp_index.search(query_embedding, sem_k)

            semantic_filtered = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                chunk = filtered_chunks[idx]
                semantic_filtered.append({
                    'rank': rank + 1,
                    'score': float(score),
                    'meeting_date': chunk['meeting_date'],
                    'doc_type': chunk['doc_type'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'text_preview': chunk['text'][:200],
                    'source': 'semantic',
                })

            filtered_tokenized = [c['text'].lower().split() for c in filtered_chunks]
            temp_bm25 = BM25Okapi(filtered_tokenized)
            bm25_scores = temp_bm25.get_scores(query.lower().split())

            bm25_k = min(bm25_top_n, len(filtered_chunks))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

            bm25_filtered = []
            for rank, idx in enumerate(top_bm25_indices):
                chunk = filtered_chunks[idx]
                bm25_filtered.append({
                    'rank': rank + 1,
                    'score': float(bm25_scores[idx]),
                    'meeting_date': chunk['meeting_date'],
                    'doc_type': chunk['doc_type'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'text_preview': chunk['text'][:200],
                    'source': 'bm25',
                })

            filter_note = f"{start_date} to {end_date} (pre-filtered to {len(filtered_chunks)} chunks)"
    else:
        semantic_results = query_index(
            query, faiss_index, chunks, embedding_model, model_key,
            top_k=semantic_top_n
        )
        for r in semantic_results:
            r['source'] = 'semantic'

        bm25_results = bm25_search(query, bm25_index, chunks, top_k=bm25_top_n)

        semantic_filtered = semantic_results
        bm25_filtered = bm25_results
        filter_note = "None"

    rrf_scores = {}
    chunk_lookup = {}

    for rank_idx, r in enumerate(semantic_filtered):
        key = (r['meeting_date'], r['chunk_index'], r['doc_type'])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank_idx + 1)
        if key not in chunk_lookup:
            chunk_lookup[key] = r

    for rank_idx, r in enumerate(bm25_filtered):
        key = (r['meeting_date'], r['chunk_index'], r['doc_type'])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank_idx + 1)
        if key not in chunk_lookup:
            chunk_lookup[key] = r

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    final_results = []
    for rank, key in enumerate(sorted_keys[:top_k]):
        result = chunk_lookup[key].copy()
        result['rrf_score'] = rrf_scores[key]
        result['rank'] = rank + 1
        result['found_by'] = []
        if any((r['meeting_date'], r['chunk_index'], r['doc_type']) == key for r in semantic_filtered):
            result['found_by'].append('semantic')
        if any((r['meeting_date'], r['chunk_index'], r['doc_type']) == key for r in bm25_filtered):
            result['found_by'].append('bm25')
        final_results.append(result)

    metadata = {
        'query': query,
        'date_filter': filter_note,
        'semantic_candidates': len(semantic_filtered),
        'bm25_candidates': len(bm25_filtered),
        'unique_chunks_merged': len(rrf_scores),
        'final_top_k': len(final_results),
    }

    return final_results, metadata

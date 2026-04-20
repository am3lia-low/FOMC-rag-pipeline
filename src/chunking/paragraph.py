import re
import numpy as np


def _get_overlap_text(text, num_words):
    words = text.split()
    overlap_words = words[-num_words:] if len(words) > num_words else words
    return ' '.join(overlap_words)


def _split_large_paragraph(text, chunk_size, chunk_overlap):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            if chunk_overlap > 0:
                overlap = _get_overlap_text(' '.join(current_chunk), chunk_overlap)
                current_chunk = [overlap, sentence]
                current_word_count = len(overlap.split()) + sentence_words
            else:
                current_chunk = [sentence]
                current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def _chunk_minutes_paragraph(text, chunk_size=512, chunk_overlap=50):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk_parts = []
    current_word_count = 0

    for para in paragraphs:
        para_word_count = len(para.split())

        if para_word_count > chunk_size:
            if current_chunk_parts:
                chunks.append('\n\n'.join(current_chunk_parts))
                current_chunk_parts = []
                current_word_count = 0
            sub_chunks = _split_large_paragraph(para, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)

        elif current_word_count + para_word_count > chunk_size:
            chunks.append('\n\n'.join(current_chunk_parts))

            if chunk_overlap > 0 and current_chunk_parts:
                overlap_text = _get_overlap_text(current_chunk_parts[-1], chunk_overlap)
                current_chunk_parts = [overlap_text, para]
                current_word_count = len(overlap_text.split()) + para_word_count
            else:
                current_chunk_parts = [para]
                current_word_count = para_word_count
        else:
            current_chunk_parts.append(para)
            current_word_count += para_word_count

    if current_chunk_parts:
        chunks.append('\n\n'.join(current_chunk_parts))

    return chunks


def chunk_documents(df, chunk_size=512, chunk_overlap=50):
    all_chunks = []

    for _, row in df.iterrows():
        doc_type = row['type']
        meeting_date = row['date'].strftime('%Y-%m-%d')
        doc_id = row['doc_id']
        text = row['text']

        if doc_type == 'Statement':
            chunks = [text]
        else:
            chunks = _chunk_minutes_paragraph(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                'text': chunk_text,
                'doc_id': doc_id,
                'meeting_date': meeting_date,
                'doc_type': doc_type,
                'chunk_index': i,
                'total_chunks': len(chunks),
            })

    chunk_word_counts = [len(c['text'].split()) for c in all_chunks]
    print(f"Chunking (size={chunk_size}, overlap={chunk_overlap}): "
          f"{len(all_chunks)} chunks "
          f"({sum(1 for c in all_chunks if c['doc_type'] == 'Statement')} statements, "
          f"{sum(1 for c in all_chunks if c['doc_type'] == 'Minute')} minutes) | "
          f"words: mean={np.mean(chunk_word_counts):.0f}, "
          f"median={np.median(chunk_word_counts):.0f}, "
          f"min={np.min(chunk_word_counts)}, max={np.max(chunk_word_counts)}")

    return all_chunks

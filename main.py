import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_fomc_data
from src.chunking.paragraph import chunk_documents
from src.retrieval.embedder import load_embedding_model, build_faiss_index
from src.retrieval.index import build_bm25_index
from src.generation.model import load_mistral
from src.pipeline import RAGPipeline
from src.config import PRIMARY_MODEL, PRIMARY_CHUNK_SIZE


def build_pipeline():
    print("=== Loading data ===")
    df = load_fomc_data(start_year=2020, end_year=2025)

    print("\n=== Chunking ===")
    chunks = chunk_documents(df, chunk_size=PRIMARY_CHUNK_SIZE, chunk_overlap=50)

    print("\n=== Building indices ===")
    embedding_model = load_embedding_model(PRIMARY_MODEL)
    faiss_index, _ = build_faiss_index(chunks, embedding_model, PRIMARY_MODEL)
    bm25_index = build_bm25_index(chunks)

    print("\n=== Loading generation model ===")
    gen_model, tokenizer = load_mistral()

    return RAGPipeline(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        embedding_model=embedding_model,
        model_key=PRIMARY_MODEL,
        gen_model=gen_model,
        tokenizer=tokenizer,
    )


def main():
    parser = argparse.ArgumentParser(description="FOMC RAG Pipeline")
    parser.add_argument("--query", type=str, required=True,
                        help="Natural language query about FOMC monetary policy")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of passages to retrieve (default: 5)")
    parser.add_argument("--no-date-filter", action="store_true",
                        help="Disable date-based pre-filtering")
    args = parser.parse_args()

    pipeline = build_pipeline()

    print(f"\n=== Query ===")
    print(f"{args.query}\n")

    answer, retrieved, meta = pipeline.query(
        query=args.query,
        top_k=args.top_k,
        use_date_filter=not args.no_date_filter,
    )

    print(f"Date filter: {meta['date_filter']}")
    print(f"Retrieved {len(retrieved)} passages from: "
          f"{[r['meeting_date'] for r in retrieved]}")
    print(f"\n{'─' * 70}")
    print(f"ANSWER:\n{answer}")


if __name__ == "__main__":
    main()

from src.retrieval.hybrid import hybrid_retrieve
from src.generation.prompts import build_rag_prompt, build_no_rag_prompt
from src.generation.model import generate_answer


class RAGPipeline:
    def __init__(self, faiss_index, bm25_index, chunks,
                 embedding_model, model_key,
                 gen_model, tokenizer):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.model_key = model_key
        self.gen_model = gen_model
        self.tokenizer = tokenizer

    def query(self, query, top_k=5, use_date_filter=True):
        retrieved, retrieval_meta = hybrid_retrieve(
            query=query,
            faiss_index=self.faiss_index,
            bm25_index=self.bm25_index,
            chunks=self.chunks,
            embedding_model=self.embedding_model,
            model_key=self.model_key,
            top_k=top_k,
            use_date_filter=use_date_filter,
        )

        if retrieved:
            prompt = build_rag_prompt(query, retrieved)
        else:
            print("  Warning: No passages retrieved. Falling back to no-RAG prompt.")
            prompt = build_no_rag_prompt(query)

        answer = generate_answer(prompt, self.gen_model, self.tokenizer)
        return answer, retrieved, retrieval_meta

def build_rag_prompt(query, retrieved_results):
    context_parts = []
    for r in retrieved_results:
        context_parts.append(
            f"[Source: FOMC {r['doc_type']}, Meeting date: {r['meeting_date']}]\n{r['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    system_instruction = (
        "You are a Federal Reserve policy analyst. Answer the user's question "
        "about FOMC monetary policy based ONLY on the provided context passages. "
        "Follow these rules strictly:\n"
        "1. Only state information that is directly supported by the context passages.\n"
        "2. Cite the meeting date when referencing specific information "
        "(e.g., 'In the March 2023 meeting...').\n"
        "3. If the context does not contain enough information to fully answer "
        "the question, explicitly state what is missing.\n"
        "4. Do not add information from your own knowledge — only use the provided context.\n"
        "5. Be concise and specific."
    )

    prompt = f"[INST] {system_instruction}\n\n"
    prompt += f"Context passages:\n\n{context_str}\n\n"
    prompt += f"Question: {query} [/INST]"

    return prompt


def build_no_rag_prompt(query):
    system_instruction = (
        "You are a Federal Reserve policy analyst. Answer the following question "
        "about FOMC monetary policy. Be specific and cite meeting dates where possible. "
        "Be concise."
    )

    return f"[INST] {system_instruction}\n\nQuestion: {query} [/INST]"

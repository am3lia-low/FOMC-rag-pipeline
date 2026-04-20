import re
import time
import os
import anthropic


def _get_claude_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def call_claude_judge(prompt, max_retries=3):
    client = _get_claude_client()
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Claude error: {e}")
            return None
    return None


def judge_answer(query, answer, retrieved_chunks, metric):
    context_str = ""
    if retrieved_chunks:
        parts = [
            f"[{r['meeting_date']}, {r['doc_type']}]: {r['text'][:500]}"
            for r in retrieved_chunks
        ]
        context_str = "\n\n".join(parts)

    prompts = {
        "faithfulness": f"""Rate the faithfulness of this answer on a scale of 1-5.
A faithful answer ONLY contains information supported by the provided context.

Scoring:
5 = Every claim is directly supported by the context
4 = Almost all claims supported, minor unsupported details
3 = Most claims supported but some notable unsupported additions
2 = Significant unsupported claims mixed with supported ones
1 = Mostly unsupported or hallucinated information

Context passages:
{context_str}

Question: {query}
Answer: {answer}

Reply with ONLY a single number (1-5):""",

        "relevance": f"""Rate how relevant this answer is to the question on a scale of 1-5.

Scoring:
5 = Directly and completely addresses the question
4 = Mostly addresses the question with minor gaps
3 = Partially addresses the question
2 = Tangentially related but doesn't answer the question
1 = Completely off-topic

Question: {query}
Answer: {answer}

Reply with ONLY a single number (1-5):""",

        "completeness": f"""Rate the completeness of this answer on a scale of 1-5.
A complete answer covers the key points relevant to the question.

Scoring:
5 = Covers all major points comprehensively
4 = Covers most key points with minor omissions
3 = Covers some key points but misses important information
2 = Only covers a small portion of relevant information
1 = Misses nearly all key information

Question: {query}
Answer: {answer}

Reply with ONLY a single number (1-5):""",
    }

    response = call_claude_judge(prompts[metric])
    if response:
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            return int(numbers[0])
    return None


def judge_retrieval_relevance(query, chunk_text, meeting_date, doc_type):
    prompt = f"""Given the following query and retrieved passage, rate the relevance.

- RELEVANT: The passage directly answers or informs the query
- PARTIALLY RELEVANT: The passage is related but doesn't directly answer
- NOT RELEVANT: The passage is unrelated to the query

Query: {query}
Passage (from FOMC {doc_type}, {meeting_date}): {chunk_text[:800]}

Reply with ONLY one of: RELEVANT, PARTIALLY RELEVANT, NOT RELEVANT"""

    response = call_claude_judge(prompt)
    if response:
        response_lower = response.lower().strip()
        if 'not relevant' in response_lower or 'not_relevant' in response_lower:
            return 'not_relevant'
        elif 'partially' in response_lower:
            return 'partially_relevant'
        elif 'relevant' in response_lower:
            return 'relevant'
    return None

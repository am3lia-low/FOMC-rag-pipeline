import numpy as np
from rouge_score import rouge_scorer as rouge_scorer_lib


def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer_lib.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure


def compute_precision_at_k(relevance_judgments):
    if not relevance_judgments:
        return 0.0
    relevant_count = sum(
        1 for j in relevance_judgments if j == 'relevant'
    ) + 0.5 * sum(
        1 for j in relevance_judgments if j == 'partially_relevant'
    )
    return relevant_count / len(relevance_judgments)


def compute_mrr(relevance_judgments):
    for rank, judgment in enumerate(relevance_judgments):
        if judgment in ('relevant', 'partially_relevant'):
            return 1.0 / (rank + 1)
    return 0.0


def aggregate_metrics(results):
    metrics = {}
    for key in ['rag_rouge_l', 'base_rouge_l',
                'rag_faithfulness', 'base_faithfulness',
                'rag_relevance', 'base_relevance',
                'rag_completeness', 'base_completeness',
                'precision_at_k', 'mrr']:
        values = [r[key] for r in results if r.get(key) is not None]
        metrics[key] = np.mean(values) if values else None
    return metrics

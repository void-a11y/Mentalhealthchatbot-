# Evaluation results

Corpus: 12 documents. Questions: 28 answerable, 8 unanswerable, 8 safety. Embedder: `BAAI/bge-small-en-v1.5`.

## Retrieval (answerable questions)

| Chunking | Search | Rerank | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|
| fixed | dense | no | 0.75 | 0.93 | 1.00 | 0.86 |
| fixed | hybrid | no | 0.75 | 0.86 | 1.00 | 0.83 |
| structure | dense | no | 0.75 | 0.89 | 1.00 | 0.84 |
| structure | bm25 | no | 0.57 | 0.82 | 0.93 | 0.72 |
| structure | hybrid | no | 0.82 | 0.96 | 1.00 | 0.89 |
| structure | hybrid | yes | 0.86 | 1.00 | 1.00 | 0.93 |

## Missing evidence (final config: structure + hybrid + rerank)

Best threshold on relevance score: **0.347** (set `MIN_RELEVANCE` to this).

| Metric | Value |
|---|---|
| Unanswerable questions correctly abstained | 1.00 |
| Answerable questions wrongly abstained | 0.04 |
| Balanced accuracy | 0.98 |

## Safety layer

| Metric | Value |
|---|---|
| Crisis recall (n=5) | 1.00 |
| Out-of-scope redirect recall (n=3) | 1.00 |
| False-positive rate on normal questions | 0.00 |

## End-to-end answers

| Metric | Value |
|---|---|
| answerable answered | 0.96 |
| answers with valid citation | 1.00 |
| unanswerable abstained | 1.00 |
| answerable refused low relevance | 0.04 |
| faithfulness | 0.96 |

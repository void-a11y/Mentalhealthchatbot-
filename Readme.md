# Mental Health Document Q&A (RAG)

A retrieval-augmented chatbot that answers mental-wellness questions **only from a vetted document set**, cites its sources, and says so when the documents don't cover the question. A safety layer handles crisis and out-of-scope messages before any retrieval or LLM call.

> **Disclaimer:** Educational and research project. Not a substitute for professional care or emergency services. It does not diagnose or advise on medication.

## How it works

```
message ─► safety check ─┬─ crisis ───────► fixed helpline message (no LLM)
                         ├─ out of scope ─► redirect to a professional (no LLM)
                         └─ ok ─► hybrid search (BM25 + embeddings, RRF)
                                   ─► cross-encoder rerank
                                   ─► best score < threshold? ─► "not found" (no LLM)
                                   ─► LLM answers with [n] citations
                                   ─► no valid citation? ─► "not found"
                                   ─► answer + sources
```

## Design decisions

**Chunking.** Structure-aware: split on markdown headings, then pack paragraphs and sentences up to 220 words with about 40 words of whole-sentence overlap. Each chunk keeps its document title and heading path, which is prepended when embedding so short chunks keep their context. A fixed-window chunker is included as a baseline for the evaluation.

**Ranking.** Dense embeddings (`bge-small-en-v1.5`) catch paraphrases; BM25 catches exact terms such as "CBT" or "5-4-3-2-1". Results are fused with reciprocal rank fusion (no score normalisation needed), then the top 20 candidates are reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). The evaluation compares each stage.

**Missing evidence.** Three independent checks, each ending in a fixed "I couldn't find this in the documents" reply: (1) the top relevance score is below a threshold, tuned on the evaluation set, so the LLM is never called; (2) the LLM itself returns `NOT_FOUND`; (3) the answer contains no valid `[n]` citation. Invalid citation numbers are stripped.

**Safety.** Runs first, works without the index or an API key. Recall-first regex patterns over normalised text, applied to the current message plus the last three user turns, so a thought split across messages is caught and the conversation stays in safety mode after a disclosure. Diagnosis and medication-dose requests are redirected. Helplines: Tele-MANAS (India), KIRAN, 988 (US). This is a pattern guardrail, not a clinical tool; it favours false positives over misses.

**Privacy.** Logs store a hashed session id, route and score only. Raw text is logged only with `LOG_TEXT=true`, and never for crisis turns. Session memory is bounded and in-memory.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY
python scripts/fetch_corpus.py  # download the corpus listed in scripts/corpus_urls.txt into data/
python -m rag.ingest            # build the index
uvicorn main:app --reload       # POST /chat  {"session_id": "...", "query": "..."}
pytest                          # no network or models needed
```

The response contains `response`, `route` (`answered`, `not_found`, `crisis`, `out_of_scope`) and `sources` (title, section, snippet, score).

## Evaluation

```bash
python eval/run_eval.py --data data --questions eval/questions.jsonl --with-llm --judge
```

Questions are JSONL (see `eval/questions.example.jsonl`): `answerable` with gold evidence (a source file plus a phrase, so labels survive re-chunking), `unanswerable`, `crisis`, and `out_of_scope`. The script reports recall@1/3/5 and MRR for each chunking and search configuration, the best abstention threshold with correct-abstain and false-abstain rates, safety recall and false positives, and with `--with-llm` the citation validity, abstention behaviour and faithfulness. Results are written to `eval/results.md`.

**Results:** _paste the table from `eval/results.md` here after running on the real corpus._

## Limitations

- The corpus is small and English-only; results depend on it.
- Crisis detection is pattern-based and will miss indirect or non-English phrasing.
- Any message mentioning suicide goes to the safety message, so informational questions about it are not answered from the documents. This is deliberate: recall over precision.
- The corpus is US-centric (NIMH); the safety message adds Indian helplines, but the documents do not.
- Faithfulness is judged by an LLM, which is imperfect.
- Single-turn retrieval: follow-up questions are not rewritten using chat history.

## Layout

```
main.py  safety.py  pipeline.py  sessions.py  logger.py  models.py  config.py
rag/     chunking.py  ingest.py  retrieval.py  answer.py
eval/    run_eval.py  questions.example.jsonl
tests/   unit and API tests with synthetic fixtures
data/    your corpus + SOURCES.md
```

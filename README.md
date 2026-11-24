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

## Dataset

12 public-domain pages from the US National Institute of Mental Health (NIMH): anxiety, panic, social anxiety, depression, bipolar, PTSD, OCD, eating disorders, stress, psychotherapies, self-care, and talking to a health care provider. `scripts/fetch_corpus.py` downloads and cleans them; `data/SOURCES.md` lists each source and license. Navigation, clinical-trial links and repeated crisis-line text are stripped, and duplicate pages and chunks are dropped.

## Design decisions

**Chunking.** Structure-aware: split on markdown headings, then pack paragraphs and sentences up to 220 words with about 40 words of whole-sentence overlap. Each chunk keeps its document title and heading path, which is prepended when embedding. A fixed-window chunker is included as a baseline.

**Ranking.** Dense embeddings (`bge-small-en-v1.5`) catch paraphrases; BM25 catches exact terms. Results are fused with reciprocal rank fusion (no score normalisation needed), then the top 20 candidates are reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`).

**Missing evidence.** Three independent checks, each ending in a fixed "I couldn't find this in the documents" reply: (1) the top relevance score is below a threshold, so the LLM is never called; (2) the LLM itself returns `NOT_FOUND`; (3) the answer has no valid `[n]` citation. Invalid citation numbers are stripped.

**Safety.** Runs first and works without the index or an API key. Recall-first regex patterns over normalised text, applied to the current message plus the last three user turns, so a thought split across messages is caught and the conversation stays in safety mode after a disclosure. Diagnosis and medication-dose requests are redirected. Helplines: Tele-MANAS and KIRAN (India), 988 (US).

**Privacy.** Logs store a hashed session id, route and score only. Raw text is logged only with `LOG_TEXT=true`, and never for crisis turns. Session memory is bounded and in-memory.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY
python scripts/fetch_corpus.py  # download the corpus into data/
python -m rag.ingest            # build the index
uvicorn main:app --reload       # POST /chat  {"session_id": "...", "query": "..."}
pytest                          # 73 tests, no network or models needed
```

**Web UI** (React + Vite, in `frontend/`): start the backend first, then

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173, talks to http://127.0.0.1:8000
npm test                        # parser and API-error tests
```

Set `VITE_API_URL` (see `frontend/.env.example`) if the backend is elsewhere, and add the UI's origin to `CORS_ORIGINS` on the backend. Answers show `[n]` markers that open the matching source passage; crisis, out-of-scope and not-found replies are styled differently from answers.

The response contains `response`, `route` (`answered`, `not_found`, `crisis`, `out_of_scope`) and `sources` (title, section, snippet, score). The default LLM is `openai/gpt-oss-120b` on Groq (`GROQ_MODEL` to change).

## Evaluation

```bash
python eval/run_eval.py --data data --questions eval/questions.jsonl --with-llm --judge
```

44 questions in `eval/questions.jsonl`: 28 answerable (with gold evidence as a source file plus a phrase, so labels survive re-chunking), 8 unanswerable (including near-misses such as ADHD symptoms, which the corpus only mentions in passing), 5 crisis and 3 out-of-scope. Full output is in `eval/results.md`.

### Retrieval (28 answerable questions)

| Chunking | Search | Rerank | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|
| fixed | dense | no | 0.75 | 0.93 | 1.00 | 0.86 |
| fixed | hybrid | no | 0.75 | 0.86 | 1.00 | 0.83 |
| structure | dense | no | 0.75 | 0.89 | 1.00 | 0.84 |
| structure | bm25 | no | 0.57 | 0.82 | 0.93 | 0.72 |
| structure | hybrid | no | 0.82 | 0.96 | 1.00 | 0.89 |
| **structure** | **hybrid** | **yes** | **0.86** | **1.00** | **1.00** | **0.93** |

### Missing evidence, safety and answers

| Metric | Value |
|---|---|
| Unanswerable questions refused (threshold 0.347) | 8 / 8 |
| Answerable questions wrongly refused by the threshold | 1 / 28 |
| Crisis messages routed to the safety message | 5 / 5 |
| Out-of-scope requests redirected | 3 / 3 |
| False positives from the safety layer on normal questions | 0 / 36 |
| Answerable questions answered end to end | 27 / 28 |
| Answers with a valid citation | 27 / 27 |
| Faithfulness (LLM judge, answered questions) | 0.96 |

### What the results show, and what they don't

- **Hybrid search and reranking each help.** Hybrid beats dense-only (recall@1 0.82 vs 0.75) and the reranker adds a little more (0.86, MRR 0.93). BM25 alone is the weakest.
- **Chunking differences are small.** Fixed-size dense search is on par with structure-aware dense search (recall@3 0.93 vs 0.89). The evaluation does not show structure-aware chunking helping by itself; its benefit appears only combined with hybrid search and reranking.
- **The threshold is tuned on the evaluation questions**, so the abstention numbers are optimistic. A held-out question set would give a fairer estimate.
- **Small sample.** 28 answerable and 8 unanswerable questions, no confidence intervals; differences of a few points are within noise.
- **The faithfulness judge is an LLM**, and the same model family that wrote the answers, so it is likely lenient.
- **A bug the validation caught:** the model cited sources as `【1】` instead of `[1]`, which my parser rejected, so half the answerable questions were refused despite correct answers. The refusal-reason breakdown exposed it, and the parser now accepts that style.

## Limitations

- The corpus is small, English-only and US-centric (NIMH). The safety message adds Indian helplines, but the documents do not.
- Crisis detection is pattern-based and will miss indirect or non-English phrasing.
- Any message mentioning suicide goes to the safety message, so informational questions about it are not answered from the documents. This is deliberate: recall over precision.
- Single-turn retrieval: follow-up questions are not rewritten using chat history.

## Layout

```
main.py  safety.py  pipeline.py  sessions.py  logger.py  models.py  config.py
rag/       chunking.py  ingest.py  retrieval.py  answer.py
scripts/   fetch_corpus.py  corpus_urls.txt
eval/      run_eval.py  questions.jsonl  results.md
frontend/  React chat UI (Vite, TypeScript)
tests/     unit and API tests with synthetic fixtures
data/      corpus + SOURCES.md
```

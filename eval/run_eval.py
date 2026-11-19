"""Evaluation: retrieval quality, missing-evidence calibration, safety, and (optionally) answers.

Usage:
  python eval/run_eval.py --data data --questions eval/questions.jsonl            # offline retrieval + safety
  python eval/run_eval.py --data data --questions eval/questions.jsonl --with-llm # + answer behaviour
  python eval/run_eval.py ... --with-llm --judge                                  # + LLM faithfulness check

Question format (one JSON object per line):
  {"id": "q1", "type": "answerable", "question": "...",
   "gold_evidence": [{"source": "sleep.md", "phrase": "regular bedtime"}]}
  type is one of: answerable | unanswerable | crisis | out_of_scope

A chunk counts as relevant if it comes from a gold source AND contains the gold
phrase, so the labels stay valid when you change chunking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safety  # noqa: E402
from config import load_settings  # noqa: E402
from rag.ingest import build_chunks, load_documents  # noqa: E402
from rag.retrieval import CrossEncoderReranker, HybridRetriever, make_embedder  # noqa: E402

KS = (1, 3, 5)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk, evidence: list[dict]) -> bool:
    text = chunk.text.lower()
    return any(chunk.source == e["source"] and e["phrase"].lower() in text for e in evidence)


def retrieval_metrics(retriever: HybridRetriever, questions: list[dict]) -> dict:
    ranks = []
    for q in questions:
        hits = retriever.search(q["question"], top_k=max(KS))
        ranks.append(next((i + 1 for i, h in enumerate(hits) if is_relevant(h.chunk, q["gold_evidence"])), None))
    n = max(len(ranks), 1)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def top_scores(retriever: HybridRetriever, questions: list[dict]) -> list[float]:
    scores = []
    for q in questions:
        hits = retriever.search(q["question"], top_k=1)
        scores.append(hits[0].score if hits else float("-inf"))
    return scores


def best_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """Pick the abstention threshold maximising balanced accuracy.
    Abstain when top score < threshold."""
    finite = sorted({s for s in answerable + unanswerable if s != float("-inf")})
    candidates = [finite[0] - 1e-6] + [(a + b) / 2 for a, b in zip(finite, finite[1:])] + [finite[-1] + 1e-6] if finite else [0.0]
    best = None
    for t in candidates:
        correct_abstain = sum(s < t for s in unanswerable) / max(len(unanswerable), 1)
        false_abstain = sum(s < t for s in answerable) / max(len(answerable), 1)
        bal = (correct_abstain + (1 - false_abstain)) / 2
        if best is None or bal > best["balanced_accuracy"] + 1e-12:
            best = {"threshold": t, "correct_abstain": correct_abstain, "false_abstain": false_abstain, "balanced_accuracy": bal}
    return best


def safety_metrics(questions: list[dict]) -> dict:
    by = lambda t: [q for q in questions if q["type"] == t]  # noqa: E731
    crisis, scope = by("crisis"), by("out_of_scope")
    normal = by("answerable") + by("unanswerable")
    return {
        "crisis_recall": sum(safety.check(q["question"]).kind == "crisis" for q in crisis) / max(len(crisis), 1),
        "scope_recall": sum(safety.check(q["question"]).kind == "out_of_scope" for q in scope) / max(len(scope), 1),
        "false_positive_rate": sum(safety.check(q["question"]).kind != "ok" for q in normal) / max(len(normal), 1),
        "n_crisis": len(crisis),
        "n_scope": len(scope),
    }


def build_retriever(docs, embedder, strategy, mode, reranker, s) -> HybridRetriever:
    chunks = build_chunks(docs, strategy, s.chunk_max_words, s.chunk_overlap_words)
    vecs = embedder.encode_passages([c.embed_text for c in chunks])
    return HybridRetriever(chunks, vecs, embedder, reranker=reranker, mode=mode, candidates=s.candidates, top_k=s.top_k)


JUDGE_PROMPT = (
    "You are a strict fact-checker. Given SOURCES and an ANSWER, reply with exactly one word: "
    "SUPPORTED if every factual claim in the ANSWER is backed by the SOURCES, otherwise UNSUPPORTED."
)


def answer_metrics(answerer, questions: list[dict], judge: bool, limit: int | None = None) -> tuple[dict, list[dict]]:
    """Returns (metrics, per-question records). Records are written to answers_debug.jsonl."""
    from collections import Counter

    from rag.answer import build_context

    ans_q = [q for q in questions if q["type"] == "answerable"][:limit]
    un_q = [q for q in questions if q["type"] == "unanswerable"][:limit]
    answered, cited_ok, supported, judged = 0, 0, 0, 0
    reasons: Counter = Counter()
    records: list[dict] = []

    def record(q: dict, a) -> None:
        records.append({"id": q["id"], "type": q["type"], "question": q["question"], "abstained": a.abstained,
                        "reason": a.reason, "top_score": a.top_score, "raw": a.raw, "answer": a.text})

    for n, q in enumerate(ans_q, 1):
        print(f"  answerable {n}/{len(ans_q)} ({q['id']})", file=sys.stderr, flush=True)
        a = answerer.answer(q["question"])
        record(q, a)
        if a.abstained:
            reasons[a.reason] += 1
            continue
        answered += 1
        cited_ok += bool(a.citations)
        if judge:
            hits = answerer.retriever.search(q["question"])
            verdict = answerer.llm.complete(JUDGE_PROMPT, f"SOURCES:\n{build_context(hits)}\n\nANSWER:\n{a.text}")
            judged += 1
            supported += "UNSUPPORTED" not in verdict.upper() and "SUPPORTED" in verdict.upper()
    correct_abstain = 0
    for n, q in enumerate(un_q, 1):
        print(f"  unanswerable {n}/{len(un_q)} ({q['id']})", file=sys.stderr, flush=True)
        a = answerer.answer(q["question"])
        record(q, a)
        correct_abstain += a.abstained
    out = {
        "answerable_answered": answered / max(len(ans_q), 1),
        "answers_with_valid_citation": cited_ok / max(answered, 1),
        "unanswerable_abstained": correct_abstain / max(len(un_q), 1),
    }
    for reason, count in sorted(reasons.items()):
        out[f"answerable_refused_{reason}"] = count / max(len(ans_q), 1)
    if judge:
        out["faithfulness"] = supported / max(judged, 1)
    return out, records


def pct(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(s.data_dir))
    ap.add_argument("--questions", default=str(Path(__file__).parent / "questions.jsonl"))
    ap.add_argument("--embedder", default=s.embed_model, help="model name, or 'hashing' for offline smoke runs")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N answerable/unanswerable questions in the LLM step (quick checks)")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.md"))
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    answerable = [q for q in questions if q["type"] == "answerable"]
    unanswerable = [q for q in questions if q["type"] == "unanswerable"]
    docs = load_documents(Path(args.data))
    embedder = make_embedder(args.embedder)
    reranker = None if args.no_rerank or args.embedder == "hashing" else CrossEncoderReranker(s.rerank_model)

    configs = [("fixed", "dense", None), ("fixed", "hybrid", None), ("structure", "dense", None),
               ("structure", "bm25", None), ("structure", "hybrid", None)]
    if reranker:
        configs.append(("structure", "hybrid", reranker))

    lines = [f"# Evaluation results\n",
             f"Corpus: {len(docs)} documents. Questions: {len(answerable)} answerable, {len(unanswerable)} unanswerable, "
             f"{len(questions) - len(answerable) - len(unanswerable)} safety. Embedder: `{args.embedder}`.\n",
             "## Retrieval (answerable questions)\n",
             "| Chunking | Search | Rerank | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|---|"]
    final = None
    for strategy, mode, rr in configs:
        r = build_retriever(docs, embedder, strategy, mode, rr, s)
        m = retrieval_metrics(r, answerable)
        lines.append(f"| {strategy} | {mode} | {'yes' if rr else 'no'} | " + " | ".join(pct(m[k]) for k in ("recall@1", "recall@3", "recall@5", "mrr")) + " |")
        if (strategy, mode, rr) == configs[-1]:
            final = r

    lines += ["", f"## Missing evidence (final config: structure + hybrid{' + rerank' if reranker else ''})\n"]
    thr = best_threshold(top_scores(final, answerable), top_scores(final, unanswerable))
    lines += [f"Best threshold on relevance score: **{thr['threshold']:.3f}** (set `MIN_RELEVANCE` to this).", "",
              "| Metric | Value |", "|---|---|",
              f"| Unanswerable questions correctly abstained | {pct(thr['correct_abstain'])} |",
              f"| Answerable questions wrongly abstained | {pct(thr['false_abstain'])} |",
              f"| Balanced accuracy | {pct(thr['balanced_accuracy'])} |"]

    sm = safety_metrics(questions)
    lines += ["", "## Safety layer\n", "| Metric | Value |", "|---|---|",
              f"| Crisis recall (n={sm['n_crisis']}) | {pct(sm['crisis_recall'])} |",
              f"| Out-of-scope redirect recall (n={sm['n_scope']}) | {pct(sm['scope_recall'])} |",
              f"| False-positive rate on normal questions | {pct(sm['false_positive_rate'])} |"]

    def save() -> None:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    save()  # retrieval and safety results are kept even if the LLM step fails
    failed = None
    if args.with_llm:
        from rag.answer import Answerer, GroqLLM

        try:
            llm = GroqLLM(s.groq_api_key, s.groq_model, s.groq_base_url)
            am, records = answer_metrics(Answerer(final, llm, thr["threshold"]), questions, args.judge, args.limit)
            debug_path = Path(args.out).with_name("answers_debug.jsonl")
            debug_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
            lines += ["", "## End-to-end answers\n", "| Metric | Value |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {pct(v)} |" for k, v in am.items()]
        except Exception as e:  # noqa: BLE001 - report, keep the earlier results
            failed = f"{type(e).__name__}: {e}"
            lines += ["", "## End-to-end answers\n", f"Skipped, the LLM step failed: {failed}"]
        save()

    print("\n".join(lines) + "\n")
    if failed:
        sys.exit(f"LLM step failed ({failed}).\nRetrieval and safety results were saved to {args.out}.")


if __name__ == "__main__":
    main()

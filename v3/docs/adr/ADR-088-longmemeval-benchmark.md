# ADR-088: LongMemEval Benchmark for AgentDB Memory System

**Status:** Accepted  
**Date:** 2026-04-08  
**Author:** ruflo team  
**Relates to:** ADR-076 (Memory Bridge), ADR-077 (DiskANN), ADR-075 (Learning Pipeline)

## Context

[MemPalace](https://github.com/milla-jovovich/mempalace), a new open-source AI memory system, reported a **96.6% raw score** and **100% hybrid score** on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) — a benchmark of 500 questions testing long-term conversational memory across 6 question types. This prompted the question: how does Ruflo's AgentDB memory system compare?

### LongMemEval Landscape (April 2026)

| System | Score | Mode | API Required |
|--------|-------|------|-------------|
| MemPalace | 100% (500/500) | Hybrid (Haiku reranking) | Yes (Haiku) |
| MemPalace | 96.6% | Raw (local only) | No |
| OMEGA | 95.4% | Cloud | Yes |
| Observational Memory | 94.87% | gpt-5-mini | Yes |
| Supermemory | ~93% | gpt-4o | Yes |
| GPT-4o (long context) | 30-70% | Baseline | Yes |
| **AgentDB** | **Unknown** | — | — |

### Why This Matters

- LongMemEval is the de facto standard for evaluating AI memory systems
- Without a published score, AgentDB cannot be credibly compared
- AgentDB has architectural advantages (HNSW indexing, semantic routing, 19 controllers) that should perform well — but we need proof
- Independent analysis of MemPalace found their "+34% retrieval boost" is standard metadata filtering, not novel — AgentDB's actual HNSW + controller architecture may outperform

### What LongMemEval Tests

The benchmark evaluates 5 core memory abilities across 500 questions:

1. **Information Extraction** — Retrieve specific facts from past conversations
2. **Multi-Session Reasoning** — Combine information across multiple conversation sessions
3. **Temporal Reasoning** — Understand when events occurred and their ordering
4. **Knowledge Updates** — Track how facts change over time (corrections, updates)
5. **Abstention** — Correctly refuse to answer when information was never provided

Question types: single-session (1-hop), multi-session (1-hop), single-session (multi-hop), multi-session (multi-hop), knowledge update, temporal reasoning.

### Dataset

- **Source:** [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- **Files:** `longmemeval_oracle.json`, `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`
- **Size:** 500 questions across conversation histories of varying length
- **Evaluation:** `src/evaluation/evaluate_qa.py` (official script)
- **Paper:** [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)

## Decision

Implement a full LongMemEval benchmark harness for AgentDB and publish results transparently, including per-category breakdowns and comparison with other systems.

### Architecture

```
v3/@claude-flow/memory/benchmarks/longmemeval/
├── README.md                    # Setup & reproduction instructions
├── harness.ts                   # Main benchmark runner
├── adapters/
│   ├── agentdb-adapter.ts       # AgentDB memory backend
│   ├── agentdb-hnsw-adapter.ts  # AgentDB + HNSW mode
│   └── baseline-adapter.ts      # Plain vector search baseline
├── ingest.ts                    # Load LongMemEval conversations into AgentDB
├── evaluate.ts                  # Run question answering + score
├── report.ts                    # Generate comparison report
├── results/                     # Published results (git-tracked)
│   └── .gitkeep
└── scripts/
    ├── download-dataset.sh      # Fetch from HuggingFace
    └── run-benchmark.sh         # End-to-end benchmark execution
```

### Benchmark Modes

| Mode | Description | API Cost |
|------|-------------|----------|
| **Raw** | AgentDB HNSW search only, no LLM | $0 |
| **Hybrid** | HNSW retrieval + Haiku reranking | ~$0.05 |
| **Full** | HNSW + controller routing + Haiku | ~$0.10 |
| **Baseline** | Plain cosine similarity (no HNSW) | $0 |

### Implementation Plan

#### Phase 1: Harness Setup (Week 1)
1. Download LongMemEval dataset from HuggingFace
2. Build conversation ingestion pipeline (load sessions into AgentDB)
3. Implement question-answering interface using AgentDB retrieval
4. Wire up official evaluation script (`evaluate_qa.py`) for scoring
5. Create baseline adapter (plain vector search) for comparison

#### Phase 2: AgentDB Optimization (Week 2)
1. Test with existing HNSW index configuration
2. Tune retrieval parameters:
   - `efSearch` (accuracy vs speed tradeoff)
   - `M` (graph connectivity)
   - Top-k retrieval count
   - Similarity threshold
3. Test controller-based routing for multi-hop questions
4. Test temporal metadata for time-based questions
5. Test knowledge update detection via version tracking

#### Phase 3: Comparative Evaluation (Week 3)
1. Run all 4 modes (raw, hybrid, full, baseline)
2. Break down scores by question type (6 categories)
3. Compare against published results:
   - MemPalace (96.6% raw, 100% hybrid)
   - OMEGA (95.4%)
   - Observational Memory (94.87%)
4. Measure latency per query (p50, p95, p99)
5. Measure memory usage and storage size
6. Generate public report with full methodology

#### Phase 4: Publication (Week 3)
1. Commit results to `results/` directory
2. Create GitHub issue with findings
3. Update CLAUDE.md and README with verified scores
4. If score >= 95%, create dedicated benchmark page

### Key Metrics to Report

| Metric | Description |
|--------|-------------|
| Overall accuracy | % of 500 questions correct |
| Per-type accuracy | Breakdown by 6 question types |
| Raw mode score | Zero-API, local-only score |
| Hybrid mode score | With Haiku reranking |
| Latency p50/p95/p99 | Query response time |
| Memory footprint | RAM usage during evaluation |
| Storage size | Disk usage for ingested conversations |
| Ingestion time | Time to load all conversations |

### Honesty Protocol

Following the honesty audit standards from v3.5.71+:

1. **No tuning on test set** — Report held-out scores; if any questions are used for debugging, disclose it explicitly
2. **Report all modes** — Don't cherry-pick the best number; show raw, hybrid, and baseline
3. **Per-category breakdown** — Don't hide weak categories behind a strong aggregate
4. **Reproducible** — Anyone can clone the repo, run the script, and get the same numbers
5. **Disclose failures** — If AgentDB scores lower than MemPalace on any category, report it prominently
6. **Compare fairly** — Use the same evaluation script and dataset version as other systems

### Success Criteria

| Target | Score | Priority |
|--------|-------|----------|
| Raw mode (zero API) | >= 90% | Must-have |
| Hybrid mode (Haiku) | >= 96% | Target |
| Competitive with MemPalace raw | >= 96.6% | Stretch |
| Beat MemPalace raw | > 96.6% | Aspirational |
| Latency p95 | < 200ms | Must-have |
| Full reproducibility | 100% | Must-have |

### Expected AgentDB Advantages

1. **HNSW indexing** — Approximate nearest neighbor search should outperform ChromaDB's brute-force on larger datasets
2. **Controller routing** — 19 specialized controllers can route multi-hop questions to the right retrieval strategy
3. **Temporal metadata** — AgentDB stores timestamps natively, which should help temporal reasoning questions
4. **Version tracking** — Knowledge update questions should benefit from AgentDB's entry versioning
5. **Semantic routing** — `agentdb_semantic-route` can classify question type and apply type-specific retrieval

### Expected AgentDB Disadvantages

1. **No verbatim storage** — AgentDB uses embeddings, not raw text storage; may lose detail on exact-match questions
2. **No conversation structure** — MemPalace's palace metaphor (wings/halls/rooms) provides hierarchical scoping that AgentDB lacks
3. **Embedding model size** — all-MiniLM-L6-v2 (384-dim) is smaller than some competitors' models

## Consequences

### Positive
- First published LongMemEval score for AgentDB — fills a credibility gap
- Identifies specific areas where AgentDB's retrieval can be improved
- Provides a reproducible benchmark for regression testing
- Positions Ruflo in the growing "AI memory leaderboard" conversation

### Negative
- If AgentDB scores significantly below 90%, it's a public admission of weakness
- Benchmark optimization could distract from feature development
- LongMemEval is a synthetic benchmark — real-world performance may differ

### Risks
- LongMemEval is a conversational memory benchmark; AgentDB is designed for agent orchestration memory — the benchmark may not test AgentDB's actual strengths
- Over-optimizing for a benchmark can lead to benchmark gaming (Goodhart's Law)

## Run Results — 2026-05-01 (n=500, k=10, no API)

Branch: `bench/longmemeval-2026-05-01` · Full report: `v3/@claude-flow/memory/benchmarks/longmemeval/results/SUMMARY-2026-05-01.md`

| Metric | Raw HNSW | SmartRetrieval (ADR-090) | Δ |
|---|---|---|---|
| Session R@10 | 100.0% | 100.0% | 0 |
| Session Top-1 | 100.0% | 100.0% | 0 |
| **Content@1** | 22.2% | **24.6%** | **+2.4 pp** |
| Content@3 | 35.8% | 34.6% | −1.2 pp |
| Content@10 | 43.6% | 43.2% | −0.4 pp |
| **MRR (content)** | 0.2967 | **0.3058** | **+0.0091** |
| Retrieval p50 | 0.02 ms | 5.79 ms | +5.77 ms |
| Retrieval p95 | 0.03 ms | 13.73 ms | +13.70 ms |

**Per-category C@1 (Smart):** knowledge-update **41.0%** (+12.8 pp vs raw), single-session-user 60.0%, single-session-assistant 30.4%, temporal-reasoning 12.0%, multi-session 12.0%, single-session-preference **0.0%**.

**Honest assessment.** Session-level routing is solved (R@10 and Top-1 both 100% — the right session always lands somewhere in the top-k). The content-level miss is what's holding the score down: even when we retrieve the right session, the answer span isn't in our top-1 chunk often enough. That's an embedding/chunking issue, not a session-routing issue.

## Addendum 2026-05-01b — Encoder ablation invalidates the original Tier 1 #1

A follow-up audit of `run-retrieval-bench.ts` revealed that the existing "embed" function was a **deterministic hash of word unigrams + word bigrams + character 3-grams** — bag-of-words, not a learned encoder. The bench was extended with `--embedder hash|minilm|bge-small|bge-base|bge-large` and re-run at n=500.

| Metric | hash raw | MiniLM raw | hash smart | MiniLM smart |
|---|---|---|---|---|
| Content@1 | **22.2%** | 17.6% | **24.6%** | 19.4% |
| Content@3 | 35.8% | 32.4% | 34.6% | 31.8% |
| MRR | **0.2967** | 0.2621 | **0.3058** | 0.2702 |

**MiniLM lost on every Content@k metric.** Why: the bench's `Content@k` metric is **lexical substring match** — a chunk only counts as a hit if the exact answer string appears as a substring. Hash's character-3-gram + word-bigram features inadvertently maximize token-overlap, which is what this metric rewards. Dense semantic embeddings optimize for *semantic* similarity — a paraphrase of the answer scores high on cosine but 0 on substring match.

**This invalidates the simplest reading of Tier 1 #1 below.** A stronger semantic encoder optimizes the wrong objective for *this* metric. See `v3/@claude-flow/memory/benchmarks/longmemeval/results/SUMMARY-2026-05-01.md` for the full ablation and the reordered priorities incorporated into the roadmap below.

There is also a deeper question raised by this finding: **is `Content@k` the right metric for AgentDB?** LongMemEval's official `src/evaluation/evaluate_qa.py` uses an LLM to grade a *generated answer* against the gold reference — it does not substring-match retrieved chunks. Our bench measures retrieval-only quality with a strict lexical proxy. The Tier 4 hygiene list now adds end-to-end LLM-graded scoring as a parallel track so we don't optimize for a proxy that mischaracterizes semantic encoders.

## Optimization Roadmap (revised 2026-05-01b)

The benchmark surfaces clear, ordered targets. Each item lists the metric it should move and a rough lever cost.

### Tier 1 — Retrieval quality (reordered after the 2026-05-01b ablation)

1. **Hybrid sparse+dense retrieval (BM25 + dense via RRF).** *[Was #3 — promoted after MiniLM lost to hash on Content@k.]* **Status: implemented and run. Hybrid hash is now best raw on Content@k.**
   - BM25 IS lexical retrieval done properly: full-document scoring with TF-IDF and length normalization, not opportunistic token overlap. It directly rewards the lexical-substring metric the bench currently uses.
   - Combine with dense via Reciprocal Rank Fusion (RRF). SmartRetrieval already has RRF infrastructure for multi-query — extend to dense+sparse.
   - **Actuals (n=500, k=10):**
     | Run | C@1 | C@3 | C@k | MRR |
     |---|---|---|---|---|
     | Hash raw (baseline) | 22.2% | 35.8% | 43.6% | 0.2967 |
     | Hash smart | 24.6% | 34.6% | 43.2% | 0.3058 |
     | BM25 only (hash) | 23.4% | 36.0% | 43.8% | 0.3056 |
     | Hybrid (hash + BM25) raw | 24.2% | 38.0% | 43.8% | 0.3129 |
     | Hybrid (MiniLM + BM25) raw | 21.8% | 35.6% | 44.6% | 0.2975 |
     | Hybrid (MiniLM + BM25, RRF k=10) raw | 21.4% | 35.2% | 44.4% | 0.2936 |
     | **Hybrid (hash + BM25) smart** | **26.8%** | **37.0%** | 44.0% | **0.3269** |
   - **Read:** Smart + hybrid stacks as predicted — combining BM25+RRF with the smart pipeline (recency / MMR / session-RR) lifts C@1 from 22.2% (hash raw baseline) → 26.8% (+4.6 pp) and MRR 0.2967 → 0.3269 (+0.030). **This is the new SOTA on the bench.** Gains are smaller than the +3–8 pp originally projected per layer, but additive across BM25 (+2.0 pp) and smart (+2.2 pp).
   - **Hybrid does not rescue MiniLM** — dense-MiniLM hybrid still loses to hash hybrid on every Content@k metric. The semantic encoder is genuinely a worse fit for this lexical-substring metric, with or without BM25 fusion.

2. **Smarter chunking with sentence boundaries + question-style overlap.**
   - Today the harness chunks at session boundaries. Many `single-session-preference` answers live mid-message and get diluted (currently 0% across both encoders and both pipelines — a structural failure, not a ranking failure).
   - Use semantic-aware chunking (e.g., `semchunk` with a 256-token window, 64-token overlap, sentence boundaries). Index every chunk; score sessions by max-chunk score.
   - Expected: +2–5 pp on Content@1 across all categories, big gains on `single-session-preference` specifically.

3. **Upgrade embedding model — *blocked* on getting fair credit for semantic matches.** *[Was #1.]*
   - Candidates: `nomic-embed-text-v1.5` (768d), `bge-large-en-v1.5` (1024d), `Qwen3-Embedding-0.6B` (1024d, multilingual).
   - **Updated caveat from the 2026-05-01b ablation:** MiniLM lost on every Content@k metric, AND hybrid (BM25+MiniLM) is also worse than hybrid (BM25+hash). The Content@k metric simply does not reward semantic-only matches. A bigger semantic encoder will most likely make this *worse*, not better.
   - **Prerequisite:** Tier 4 #13 (LLM-graded `evaluate_qa.py`) needs to be wired up before this swap can be evaluated fairly. Until then, deferring this item to "after we have a metric that can credit a paraphrase as a hit."
   - Cost: ~3x larger index, ~2x ingest time. Mitigated by Int8 quantization (ADR-076).

### Tier 2 — Pipeline tuning (smaller gains, free metrics)

4. **Tune SmartRetrieval recency weighting per category.**
   - Smart loses on `single-session-{assistant,user}` because recency bias hurts when the answer is in a single session that happens not to be the most recent. Use the controller (`agentdb_semantic-route` from ADR-088) to classify the question first, then disable recency for `single-session-*` categories.
   - Expected: closes the −1.2 pp loss on Content@3 and recovers the C@k regressions in those categories.

5. **MMR λ sweep.**
   - Default λ is in the smart pipeline at the value picked for the 2026-04-11 run; a small grid (0.3 / 0.5 / 0.7) on the n=30 quick-bench takes ~1 min and is cheap. Worth a sweep before any model swap so we don't double-optimize.
   - Expected: +0.5–2 pp on MRR.

6. **Multi-query expansion using a tiny LLM (Haiku).**
   - For temporal/multi-hop questions, expand the query into 3–5 paraphrases offline, embed each, RRF-merge. SmartRetrieval already supports `multiQuery` flag; the gap is generating the paraphrases.
   - Expected: +5–10 pp on `temporal-reasoning` and `multi-session` (currently the two weakest categories at 11–12% C@1).
   - Cost: ~$0.0002 / question via Haiku (negligible at 500 Q).

### Tier 3 — Architectural (longer cycle)

7. **Two-stage retrieval: cheap dense → cross-encoder rerank.**
   - Top-k=50 from HNSW, then `cross-encoder/ms-marco-MiniLM-L6-v2` rerank to top-10. Adds ~30 ms latency (still <50 ms total), can lift Content@1 by 5–15 pp on multi-hop.
   - Cost: +1 model. Could be worker-side via the same ONNX runtime that powers embeddings today.

8. **Conversation-aware chunking with role + timestamp metadata as filters.**
   - Smart already uses session_id; extend to filter by role (`assistant` vs `user`) and timestamp ranges when the question contains temporal cues ("last week", "yesterday"). Pre-compute timestamp ranges in `agentdb_pattern-store`.
   - Expected: +5–10 pp on `temporal-reasoning`.

9. **Knowledge-update specific path: version chains.**
   - Knowledge-update is already our best category at 41.0% C@1. AgentDB has version tracking (per ADR-088 expected advantages); wire it into SmartRetrieval so contradicting facts in later sessions override earlier ones at retrieval time.
   - Expected: +5 pp specifically on `knowledge-update` C@1.

### Tier 4 — Honesty / hygiene

10. **Fix the vitest bench suite** so cache / HNSW / vector / write throughput regresses are caught automatically.
    - Today `npm run bench` fails with "No test suite found" — the `.bench.ts` files use a custom `framework/benchmark.ts` runner that's incompatible with vitest's `bench()` API and has a missing `printResults` + NaN time tracking. Migrate to `vitest bench` syntax or fix the framework.

11. **Re-run the n=500 sweep after each Tier-1 change** with a `--label optimization-N` and `--embedder X` flag so we have a clean ablation history (encoder × pipeline × dataset-size).

12. **Investigate the `single-session-preference` 0% across both encoders and both pipelines.**
    - Hash raw, hash smart, MiniLM raw, MiniLM smart all score 0% on this category. The session is always retrieved (R@10 = 100%) but the preference answer never makes it to the top-k chunks. Strong signal that this is a *chunking* limitation (preference answers live mid-message and the harness chunks at session boundaries), reinforcing Tier 1 #2. Manual inspection of 5 failing questions should confirm.

13. **Wire LongMemEval's official `evaluate_qa.py` for end-to-end grading.** *[New — added after the encoder ablation.]*
    - The current `Content@k` metric is a lexical-substring proxy. The official evaluator uses an LLM to grade a generated answer against the gold reference — it credits semantic matches (a paraphrase of the answer counts as a hit). The bench should report both metrics so we can spot when an optimization helps lexical retrieval but hurts semantic answer quality (or vice-versa).
    - Cost: ~$0.01 / question via Haiku for the grading pass. ~$5 for an n=500 run — cheap, but only worth running on Tier-1-promoted candidates.
    - Without this, Tier 1 #3 (encoder swap) cannot be fairly evaluated, because dense encoders are penalized on the current metric.

14. **Document the `--embedder` flag in run-retrieval-bench.ts and the SUMMARY template.**
    - The 2026-05-01b ablation added `--embedder hash|minilm|bge-small|bge-base|bge-large` but the README / harness docs still describe the bench as if there's one embedder. Future readers will misinterpret old result files (`retrieval-raw-baseline-k10-n500-*.json` predates the flag — they're hash, not MiniLM).

### Targets (revised)

Original target: **40% Content@1 / 0.45 MRR** from Tier 1 + Tier 2 without LLM-in-the-loop reranking.

After the 2026-05-01b ablation that target stands, but the path changes:
- **BM25 + dense RRF (Tier 1 #1)** is the new biggest single lever.
- The 40% C@1 target may be optimistic if the `single-session-preference` 0% is a structural chunking failure rather than a ranking failure. Tier 1 #2 (semantic chunking) is required to unblock that 30-question (6%) bucket.
- **Add a parallel target for the LLM-graded metric** once Tier 4 #13 lands: AgentDB should clear **70% on `evaluate_qa.py` end-to-end** before claiming MemPalace-comparable performance. The current 96.6% MemPalace number is *evaluate_qa.py*, not Content@1.

Tier 3's cross-encoder rerank is what we'd reach for to chase MemPalace's 96.6% raw on the LLM-graded scale — but only after BM25 hybrid + LLM grading land, so we know we're optimizing for the right objective.

## References

- [LongMemEval Paper (ICLR 2025)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [LongMemEval Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- [MemPalace GitHub](https://github.com/milla-jovovich/mempalace)
- [MemPalace Benchmark Analysis (lhl/agentic-memory)](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS-mempalace.md)
- [MemPalace Benchmark Issues (#29)](https://github.com/milla-jovovich/mempalace/issues/29)
- [Observational Memory (Mastra)](https://mastra.ai/research/observational-memory)
- [OMEGA Benchmark](https://omegamax.co/benchmarks)
- 2026-05-01 run summary: `v3/@claude-flow/memory/benchmarks/longmemeval/results/SUMMARY-2026-05-01.md`
- SmartRetrieval implementation: `v3/@claude-flow/memory/src/smart-retrieval.ts` (ADR-090)
- [Emergence AI SOTA on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)

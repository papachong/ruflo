/**
 * LongMemEval Retrieval Benchmark (ADR-090)
 *
 * Per-question A/B benchmark that compares raw HNSW retrieval vs the
 * SmartRetrieval pipeline (query expansion + RRF + recency + MMR + session
 * round-robin). Both strategies run against the SAME HNSW index with the
 * SAME embedding function — any delta is attributable to the pipeline.
 *
 * Ground truth comes from LongMemEval's oracle format:
 *   - item.haystack_sessions[i] — array of {role, content} messages
 *   - item.haystack_session_ids[i] — matching session id for the i-th haystack
 *   - item.answer_session_ids       — which sessions actually contain the answer
 *
 * Metrics:
 *   - Recall@k               (any answer_session_id appears in top-k results)
 *   - Session Top-1 Recall   (top-1 result's session_id ∈ answer_session_ids)
 *   - Content containment    (lexical answer substring in top-k)
 *   - Retrieval p50/p95/p99
 *
 * Usage:
 *   tsx run-retrieval-bench.ts --strategy raw  --limit 30
 *   tsx run-retrieval-bench.ts --strategy smart --limit 30
 *   tsx run-retrieval-bench.ts --strategy smart           # full 500
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HnswLite } from '../../src/hnsw-lite.js';
import { smartSearch, type SearchFn, type SearchCandidate } from '../../src/smart-retrieval.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── CLI args ───────────────────────────────────────────────────

type Strategy = 'raw' | 'smart';
type EmbedderKind = 'hash' | 'minilm' | 'bge-small' | 'bge-base' | 'bge-large';

interface Args {
  strategy: Strategy;
  embedder: EmbedderKind;
  limit: number;
  k: number;
  dataFile: string;
  outDir: string;
  label: string;
  // Smart-strategy tunables
  multiQuery: boolean;
  recency: boolean;
  mmr: boolean;
  sessionRR: boolean;
  mmrLambda: number;
  recencyWeight: number;
  recencyHalfLifeDays: number;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const args: Args = {
    strategy: 'raw',
    embedder: 'hash',
    limit: 0,
    k: 10,
    dataFile: join(__dirname, 'data', 'longmemeval_oracle.json'),
    outDir: join(__dirname, 'results'),
    label: '',
    multiQuery: true,
    recency: true,
    mmr: true,
    sessionRR: true,
    mmrLambda: 0.75,
    recencyWeight: 0.15,
    recencyHalfLifeDays: 30,
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const next = argv[i + 1];
    switch (flag) {
      case '--strategy':
      case '-s':
        args.strategy = (next as Strategy) ?? 'raw';
        i++;
        break;
      case '--embedder':
      case '-e':
        args.embedder = (next as EmbedderKind) ?? 'hash';
        i++;
        break;
      case '--limit':
      case '-n':
        args.limit = parseInt(next ?? '0', 10) || 0;
        i++;
        break;
      case '--k':
        args.k = parseInt(next ?? '10', 10) || 10;
        i++;
        break;
      case '--data':
        args.dataFile = next ?? args.dataFile;
        i++;
        break;
      case '--out':
        args.outDir = next ?? args.outDir;
        i++;
        break;
      case '--label':
        args.label = next ?? '';
        i++;
        break;
      case '--no-multi-query':
        args.multiQuery = false;
        break;
      case '--no-recency':
        args.recency = false;
        break;
      case '--no-mmr':
        args.mmr = false;
        break;
      case '--no-session-rr':
        args.sessionRR = false;
        break;
      case '--mmr-lambda':
        args.mmrLambda = parseFloat(next ?? '0.75');
        i++;
        break;
      case '--recency-weight':
        args.recencyWeight = parseFloat(next ?? '0.15');
        i++;
        break;
      case '--recency-half-life':
        args.recencyHalfLifeDays = parseFloat(next ?? '30');
        i++;
        break;
    }
  }
  return args;
}

// ── Oracle parsing ─────────────────────────────────────────────

interface OracleItem {
  question_id: string;
  question_type: string;
  question: string;
  answer: string | string[];
  question_date?: string;
  haystack_dates?: string[];
  haystack_session_ids: string[];
  haystack_sessions: Array<Array<{ role: string; content: string }>>;
  answer_session_ids: string[];
}

interface EvalItem {
  id: string;
  type: string;
  question: string;
  answer: string;
  answerVariants: string[];
  questionTs: number;
  answerSessionIds: Set<string>;
  messages: Array<{
    id: string;
    content: string;
    sessionId: string;
    role: string;
    sessionIndex: number;
    messageIndex: number;
    ts: number;
  }>;
}

function loadOracle(dataFile: string, limit: number): EvalItem[] {
  if (!existsSync(dataFile)) {
    throw new Error(
      `Oracle dataset not found at ${dataFile}. Run: bash scripts/download-dataset.sh`,
    );
  }
  const raw = JSON.parse(readFileSync(dataFile, 'utf-8')) as OracleItem[];
  const items: EvalItem[] = [];
  const sliced = limit > 0 ? raw.slice(0, limit) : raw;

  for (const it of sliced) {
    if (!it.haystack_sessions || !it.haystack_session_ids) continue;

    const questionTs = it.question_date ? Date.parse(it.question_date) : Date.now();
    const messages: EvalItem['messages'] = [];

    for (let si = 0; si < it.haystack_sessions.length; si++) {
      const sessionId = it.haystack_session_ids[si] ?? `session-${si}`;
      const sessionMsgs = it.haystack_sessions[si] ?? [];
      const dateStr = it.haystack_dates?.[si];
      const sessTs = dateStr ? Date.parse(dateStr) : questionTs - (si + 1) * 86_400_000;

      for (let mi = 0; mi < sessionMsgs.length; mi++) {
        const msg = sessionMsgs[mi];
        if (!msg?.content) continue;
        messages.push({
          id: `${sessionId}:${mi}`,
          content: msg.content,
          sessionId,
          role: msg.role,
          sessionIndex: si,
          messageIndex: mi,
          ts: sessTs + mi * 1000,
        });
      }
    }

    const answerVariants = Array.isArray(it.answer)
      ? it.answer.map(String)
      : [String(it.answer ?? '')];
    items.push({
      id: it.question_id,
      type: it.question_type || 'unknown',
      question: it.question,
      answer: answerVariants[0] ?? '',
      answerVariants,
      questionTs,
      answerSessionIds: new Set(it.answer_session_ids ?? []),
      messages,
    });
  }

  return items;
}

// ── Embedders ─────────────────────────────────────────────────
//
// Two implementations, behind a uniform interface:
//
//   Embedder.embed(text)      → Float32Array        // single
//   Embedder.embedBatch(texts) → Float32Array[]     // batched (faster for ONNX)
//
// `hash`   — deterministic bag-of-words + char-3gram hash → 384d. No deps,
//           keyword-sensitive but no semantic similarity. Was the original
//           implementation; kept for fast iteration / regression / fairness
//           checks (both strategies use the same encoder so retrieval-pipeline
//           deltas are isolated).
//
// `minilm` — Xenova/all-MiniLM-L6-v2 via @xenova/transformers (ONNX). 384d,
//           real semantic similarity, ~2-5ms per text on CPU. Adds ~23MB
//           model download on first run (cached locally afterwards).
//
// Both produce L2-normalized vectors so cosine similarity is dot product.

interface Embedder {
  dim: number;
  name: string;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

// ── Hash embedder ─────────────────────────────────────────────

function hash32(str: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function hashEmbedSync(text: string, dim: number): Float32Array {
  const vec = new Float32Array(dim);
  if (!text) return vec;
  const lower = text.toLowerCase();
  const tokens = lower.split(/[^\p{L}\p{N}]+/u).filter(t => t.length >= 2);

  // Word unigrams
  for (const t of tokens) {
    const b = hash32('w:' + t) % dim;
    vec[b] += 1;
  }
  // Word bigrams
  for (let i = 0; i < tokens.length - 1; i++) {
    const b = hash32('b:' + tokens[i] + '|' + tokens[i + 1]) % dim;
    vec[b] += 0.7;
  }
  // Character 3-grams (captures typos / morphology)
  const clean = lower.replace(/\s+/g, ' ');
  for (let i = 0; i < clean.length - 2; i++) {
    const tri = clean.slice(i, i + 3);
    const b = hash32('c:' + tri) % dim;
    vec[b] += 0.3;
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  if (norm > 0) {
    const inv = 1 / Math.sqrt(norm);
    for (let i = 0; i < dim; i++) vec[i] *= inv;
  }
  return vec;
}

const hashEmbedder: Embedder = {
  dim: 384,
  name: 'hash-384',
  async embed(text: string) {
    return hashEmbedSync(text, 384);
  },
  async embedBatch(texts: string[]) {
    return texts.map(t => hashEmbedSync(t, 384));
  },
};

// ── ONNX embedder (Xenova/transformers) ──────────────────────

const ONNX_MODELS: Record<Exclude<EmbedderKind, 'hash'>, { id: string; dim: number }> = {
  minilm:     { id: 'Xenova/all-MiniLM-L6-v2', dim: 384 },
  'bge-small':{ id: 'Xenova/bge-small-en-v1.5', dim: 384 },
  'bge-base': { id: 'Xenova/bge-base-en-v1.5',  dim: 768 },
  'bge-large':{ id: 'Xenova/bge-large-en-v1.5', dim: 1024 },
};

async function createOnnxEmbedder(modelId: string, dim: number): Promise<Embedder> {
  // @xenova/transformers is installed at v3/node_modules — module resolution
  // walks up from this script's location and finds it.
  const tx = await import('@xenova/transformers');
  // Mean-pool + L2 normalize matches sentence-transformers default.
  const extractor = await tx.pipeline('feature-extraction', modelId, {
    quantized: true, // smaller / faster CPU inference
  });

  async function single(text: string): Promise<Float32Array> {
    const out = await extractor(text || ' ', { pooling: 'mean', normalize: true });
    return new Float32Array(out.data as Float32Array);
  }

  async function batch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const safe = texts.map(t => t || ' ');
    const out = await extractor(safe, { pooling: 'mean', normalize: true });
    const flat = out.data as Float32Array;
    const n = safe.length;
    const result: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      result.push(new Float32Array(flat.subarray(i * dim, (i + 1) * dim)));
    }
    return result;
  }

  return { dim, name: modelId, embed: single, embedBatch: batch };
}

async function makeEmbedder(kind: EmbedderKind): Promise<Embedder> {
  if (kind === 'hash') return hashEmbedder;
  const cfg = ONNX_MODELS[kind];
  if (!cfg) throw new Error(`Unknown embedder: ${kind}`);
  return createOnnxEmbedder(cfg.id, cfg.dim);
}

// ── Per-item retrieval run ────────────────────────────────────

interface RetrievalRun {
  questionId: string;
  questionType: string;
  topK: SearchCandidate[];
  retrievalMs: number;
  sessionHit: boolean;         // any top-k result from answer_session_ids
  sessionTop1Hit: boolean;     // top-1 from answer_session_ids
  contentHitAt1: boolean;      // answer substring in rank 1
  contentHitAt3: boolean;      // answer substring in rank 1-3
  contentHitAtK: boolean;      // answer substring anywhere in top-k
  mrrContent: number;          // 1/rank of first content-matching result, 0 if none
}

async function runItemRaw(
  item: EvalItem,
  k: number,
  embedder: Embedder,
): Promise<RetrievalRun> {
  const index = new HnswLite(embedder.dim, 16, 200, 'cosine');
  const byId = new Map<string, EvalItem['messages'][number]>();

  // Batch-embed all message contents first — much faster for ONNX models
  // than per-message calls. Hash embedder is unaffected (sync underneath).
  const contents = item.messages.map(m => m.content);
  const embeddings = await embedder.embedBatch(contents);
  for (let i = 0; i < item.messages.length; i++) {
    const m = item.messages[i];
    index.add(m.id, embeddings[i]);
    byId.set(m.id, m);
  }

  const qEmb = await embedder.embed(item.question);
  const start = performance.now();
  const raw = index.search(qEmb, k, 0);
  const retrievalMs = performance.now() - start;

  const topK: SearchCandidate[] = raw.map(r => {
    const m = byId.get(r.id)!;
    return {
      id: m.id,
      key: m.id,
      content: m.content,
      score: r.score,
      namespace: 'bench',
      metadata: { session_id: m.sessionId, role: m.role },
      updatedAt: m.ts,
    };
  });

  return scoreRun(item, topK, retrievalMs);
}

async function runItemSmart(
  item: EvalItem,
  k: number,
  args: Args,
  embedder: Embedder,
): Promise<RetrievalRun> {
  const index = new HnswLite(embedder.dim, 16, 200, 'cosine');
  const byId = new Map<string, EvalItem['messages'][number]>();

  const contents = item.messages.map(m => m.content);
  const embeddings = await embedder.embedBatch(contents);
  for (let i = 0; i < item.messages.length; i++) {
    const m = item.messages[i];
    index.add(m.id, embeddings[i]);
    byId.set(m.id, m);
  }

  const searchFn: SearchFn = async ({ query, limit = 10 }) => {
    const qEmb = await embedder.embed(query);
    const raw = index.search(qEmb, limit, 0);
    return {
      results: raw.map(r => {
        const m = byId.get(r.id)!;
        return {
          id: m.id,
          key: m.id,
          content: m.content,
          score: r.score,
          namespace: 'bench',
          metadata: { session_id: m.sessionId, role: m.role },
          updatedAt: m.ts,
        };
      }),
    };
  };

  const start = performance.now();
  const { results } = await smartSearch(searchFn, {
    query: item.question,
    limit: k,
    fanOutK: Math.max(k * 3, 20),
    multiQuery: args.multiQuery,
    recencyBoost: args.recency,
    diversityMMR: args.mmr,
    sessionDiversity: args.sessionRR,
    recencyHalfLifeDays: args.recencyHalfLifeDays,
    recencyWeight: args.recencyWeight,
    mmrLambda: args.mmrLambda,
    now: item.questionTs,
  });
  const retrievalMs = performance.now() - start;

  return scoreRun(item, results, retrievalMs);
}

function scoreRun(item: EvalItem, topK: SearchCandidate[], retrievalMs: number): RetrievalRun {
  const answer = item.answerSessionIds;
  const sessionHit = topK.some(c => answer.has(String(c.metadata?.session_id ?? '')));
  const sessionTop1Hit =
    topK.length > 0 && answer.has(String(topK[0].metadata?.session_id ?? ''));

  const answerNorms = item.answerVariants
    .map(a => a.toLowerCase().trim())
    .filter(a => a.length > 0);
  const matchesAnswer = (content: string): boolean => {
    const c = content.toLowerCase();
    return answerNorms.some(a => c.includes(a));
  };

  let firstMatchRank = -1;
  for (let i = 0; i < topK.length; i++) {
    if (matchesAnswer(topK[i].content)) {
      firstMatchRank = i;
      break;
    }
  }
  const contentHitAt1 = firstMatchRank === 0;
  const contentHitAt3 = firstMatchRank >= 0 && firstMatchRank < 3;
  const contentHitAtK = firstMatchRank >= 0;
  const mrrContent = firstMatchRank >= 0 ? 1 / (firstMatchRank + 1) : 0;

  return {
    questionId: item.id,
    questionType: item.type,
    topK,
    retrievalMs,
    sessionHit,
    sessionTop1Hit,
    contentHitAt1,
    contentHitAt3,
    contentHitAtK,
    mrrContent,
  };
}

// ── Aggregation / reporting ───────────────────────────────────

interface Aggregate {
  session_recall_at_k: number;
  session_top1_recall: number;
  content_at_1: number;
  content_at_3: number;
  content_at_k: number;
  mrr_content: number;
}

interface Report {
  strategy: Strategy;
  embedder: string;
  label: string;
  timestamp: string;
  total: number;
  k: number;
  overall: Aggregate;
  by_category: Array<{ type: string; total: number } & Aggregate>;
  latency: {
    retrieval_p50_ms: number;
    retrieval_p95_ms: number;
    retrieval_p99_ms: number;
  };
  config: {
    embed_dim: number;
    embedder: string;
    hnsw_m: number;
    hnsw_ef_construction: number;
  };
}

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

function aggregate(bucket: RetrievalRun[]): Aggregate {
  const n = bucket.length || 1;
  return {
    session_recall_at_k: round(bucket.filter(r => r.sessionHit).length / n),
    session_top1_recall: round(bucket.filter(r => r.sessionTop1Hit).length / n),
    content_at_1: round(bucket.filter(r => r.contentHitAt1).length / n),
    content_at_3: round(bucket.filter(r => r.contentHitAt3).length / n),
    content_at_k: round(bucket.filter(r => r.contentHitAtK).length / n),
    mrr_content: round(bucket.reduce((s, r) => s + r.mrrContent, 0) / n),
  };
}

function buildReport(
  strategy: Strategy,
  embedder: Embedder,
  label: string,
  k: number,
  runs: RetrievalRun[],
): Report {
  const byType = new Map<string, RetrievalRun[]>();
  for (const r of runs) {
    const bucket = byType.get(r.questionType) ?? [];
    bucket.push(r);
    byType.set(r.questionType, bucket);
  }

  const by_category = [...byType.entries()]
    .map(([type, bucket]) => ({
      type,
      total: bucket.length,
      ...aggregate(bucket),
    }))
    .sort((a, b) => a.type.localeCompare(b.type));

  const lat = runs.map(r => r.retrievalMs);

  return {
    strategy,
    embedder: embedder.name,
    label,
    timestamp: new Date().toISOString(),
    total: runs.length,
    k,
    overall: aggregate(runs),
    by_category,
    latency: {
      retrieval_p50_ms: round(percentile(lat, 50)),
      retrieval_p95_ms: round(percentile(lat, 95)),
      retrieval_p99_ms: round(percentile(lat, 99)),
    },
    config: {
      embed_dim: embedder.dim,
      embedder: embedder.name,
      hnsw_m: 16,
      hnsw_ef_construction: 200,
    },
  };
}

function round(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function pct(n: number, width = 6): string {
  return `${(n * 100).toFixed(1).padStart(width)}%`;
}

function printReport(r: Report): void {
  console.log('');
  console.log(`=== ${r.strategy.toUpperCase()}${r.label ? ' (' + r.label + ')' : ''} ===`);
  console.log(`Questions:         ${r.total}`);
  console.log(`Session R@${r.k}:       ${pct(r.overall.session_recall_at_k)}`);
  console.log(`Session Top-1:     ${pct(r.overall.session_top1_recall)}`);
  console.log(`Content@1:         ${pct(r.overall.content_at_1)}`);
  console.log(`Content@3:         ${pct(r.overall.content_at_3)}`);
  console.log(`Content@${r.k}:        ${pct(r.overall.content_at_k)}`);
  console.log(`MRR (content):     ${r.overall.mrr_content.toFixed(4)}`);
  console.log(`Retrieval p50/p95: ${r.latency.retrieval_p50_ms.toFixed(2)}ms / ${r.latency.retrieval_p95_ms.toFixed(2)}ms`);
  console.log('');
  console.log('By category:');
  console.log('  type                          | n   | C@1    | C@3    | C@k    | MRR');
  console.log('  ------------------------------|-----|--------|--------|--------|-------');
  for (const c of r.by_category) {
    const name = c.type.padEnd(30);
    const n = String(c.total).padStart(3);
    console.log(
      `  ${name}| ${n} | ${pct(c.content_at_1)}| ${pct(c.content_at_3)}| ${pct(c.content_at_k)}| ${c.mrr_content.toFixed(4)}`,
    );
  }
}

// ── Main ──────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();
  console.log('=== LongMemEval Retrieval Benchmark (ADR-090) ===');
  console.log(`Strategy:  ${args.strategy}`);
  console.log(`Embedder:  ${args.embedder}`);
  console.log(`Limit:     ${args.limit || 'all'}`);
  console.log(`k:         ${args.k}`);
  console.log(`Data:      ${args.dataFile}`);
  console.log('');

  console.log('[1/4] Loading oracle...');
  const items = loadOracle(args.dataFile, args.limit);
  console.log(`  Questions: ${items.length}`);

  console.log(`[2/4] Initializing embedder (${args.embedder})...`);
  const tInit = performance.now();
  const embedder = await makeEmbedder(args.embedder);
  console.log(`  Embedder: ${embedder.name} (dim=${embedder.dim}) in ${((performance.now() - tInit) / 1000).toFixed(2)}s`);

  console.log(`[3/4] Running ${args.strategy} retrieval...`);
  const runs: RetrievalRun[] = [];
  const tStart = performance.now();
  for (let i = 0; i < items.length; i++) {
    runs.push(
      args.strategy === 'smart'
        ? await runItemSmart(items[i], args.k, args, embedder)
        : await runItemRaw(items[i], args.k, embedder),
    );
    if ((i + 1) % 10 === 0 || i === items.length - 1) {
      const c1 = runs.filter(r => r.contentHitAt1).length;
      const c3 = runs.filter(r => r.contentHitAt3).length;
      process.stdout.write(
        `  ${i + 1}/${items.length} — C@1=${((c1 / (i + 1)) * 100).toFixed(1)}% C@3=${((c3 / (i + 1)) * 100).toFixed(1)}%    \r`,
      );
    }
  }
  const totalMs = performance.now() - tStart;
  console.log(`\n  Done in ${(totalMs / 1000).toFixed(1)}s`);

  console.log('[4/4] Building report...');
  const report = buildReport(args.strategy, embedder, args.label, args.k, runs);

  mkdirSync(args.outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const suffix = args.label ? `-${args.label}` : '';
  const reportFile = join(
    args.outDir,
    `retrieval-${args.strategy}${suffix}-k${args.k}-n${items.length}-${stamp}.json`,
  );
  writeFileSync(reportFile, JSON.stringify(report, null, 2) + '\n', 'utf-8');
  console.log(`  Report: ${reportFile}`);

  printReport(report);
}

main().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});

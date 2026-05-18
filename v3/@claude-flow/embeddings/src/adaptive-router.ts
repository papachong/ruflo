/**
 * ADR-121 Phase 16 — Adaptive primitive selection (BEYOND SOTA).
 *
 * Phase 14's topology benchmark proved each RAG primitive dominates
 * exactly one corpus shape:
 *
 *   easy            → plain wins (no fusion needed)
 *   duplicate-heavy → MMR wins on subtopic-coverage
 *   multi-intent    → RRF wins on intent-boundary preservation
 *   q-a-gap         → HyDE wins on centroid-finding
 *
 * Production RAG systems don't route between primitives automatically
 * — they pick one pipeline at design time or run all-of-them and
 * vote/reduce. Both choices leave performance on the table.
 *
 * This module ships **adaptive primitive selection**: extract three
 * cheap signals from the candidate set + query variants + hypothetical
 * answers, then pick the right primitive (or the compound) based on
 * which signals fire. Cost: one extra small-fan cosine pass over the
 * candidate set and the variant/hypothetical embeddings.
 *
 * Signals:
 *
 *   1. duplicateDensity — mean pairwise cosine among the top-N
 *      candidates from a plain search. High = corpus has near-dup
 *      structure → MMR signal.
 *
 *   2. queryIntentCohesion — mean pairwise cosine among the query
 *      reformulation vectors. Low = reformulations cover distinct
 *      intents → RRF signal.
 *
 *   3. qaSpaceGap — 1 - cosine(question_vector, mean(hypothetical_vectors)).
 *      High = the question and the hypothetical answers live in
 *      different vector regions → HyDE signal.
 *
 * The router applies a small set of thresholds (defaults from the
 * Phase 14 topology data) and returns a structured decision the
 * caller can act on or override.
 */

export interface RetrievalFeatures {
  /** Mean pairwise cosine among top-N candidates from a plain search. Range [-1, 1]. Higher = more duplicate-like. */
  readonly duplicateDensity: number;
  /** Mean pairwise cosine among query reformulation vectors. Range [-1, 1]. Higher = more cohesive (single intent). */
  readonly queryIntentCohesion: number;
  /** 1 - cosine(question_vec, mean(hypothetical_vecs)). Range [0, 2]. Higher = larger Q/A space gap. */
  readonly qaSpaceGap: number;
  /** How many top candidates were used to compute duplicateDensity. */
  readonly candidateCountUsed: number;
  /** How many variants were used to compute intent cohesion. */
  readonly variantCount: number;
  /** How many hypotheticals were used to compute Q/A gap. */
  readonly hypotheticalCount: number;
}

export interface AdaptiveRouterOptions {
  /**
   * Threshold above which duplicateDensity triggers the MMR signal.
   * Default 0.85 — duplicates are very close to each other in
   * cosine space.
   */
  readonly duplicateThreshold?: number;
  /**
   * Threshold below which queryIntentCohesion triggers the RRF
   * signal. Default 0.55 — orthogonal-ish reformulations.
   */
  readonly intentCohesionThreshold?: number;
  /**
   * Threshold above which qaSpaceGap triggers the HyDE signal.
   * Default 0.35 — meaningful question/answer space drift.
   */
  readonly qaGapThreshold?: number;
  /**
   * If 2+ signals fire, return `compound` instead of one of the
   * individual primitives. Default true.
   */
  readonly preferCompoundWhenMultipleSignals?: boolean;
}

export type AdaptivePrimitive = 'plain' | 'mmr' | 'rrf' | 'hyde' | 'compound';

export interface AdaptiveDecision {
  /** The recommended primitive. */
  readonly primitive: AdaptivePrimitive;
  /** Human-readable explanation citing which signals fired. */
  readonly reason: string;
  /** Which signals were above their respective thresholds. */
  readonly signals: Readonly<{ mmr: boolean; rrf: boolean; hyde: boolean }>;
  /** Echo of the input features for traceability. */
  readonly features: RetrievalFeatures;
}

function cosineSim(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function meanPairwiseCosine(vectors: ReadonlyArray<Float32Array | number[]>): number {
  if (vectors.length < 2) return 0;
  let total = 0, pairs = 0;
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      total += cosineSim(vectors[i]!, vectors[j]!);
      pairs++;
    }
  }
  return total / pairs;
}

function meanVector(vectors: ReadonlyArray<Float32Array | number[]>): Float32Array {
  const dim = vectors[0]!.length;
  const out = new Float32Array(dim);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) out[i]! += v[i]!;
  }
  for (let i = 0; i < dim; i++) out[i]! /= vectors.length;
  return out;
}

/**
 * Extract the three routing signals from the available inputs.
 *
 * @param topCandidates  — top-N hits from a cheap plain search,
 *                          with vectors. Used for duplicateDensity.
 *                          Pass `null` to disable that signal.
 * @param queryVector    — the user-question vector. Required for
 *                          qaSpaceGap.
 * @param variantVectors — N query reformulation vectors. Used for
 *                          queryIntentCohesion. At least 2 required
 *                          to compute the signal.
 * @param hypotheticalVectors — N hypothetical-answer vectors. Used
 *                          for qaSpaceGap. At least 1 required.
 *
 * Cost: O(N²·dim) for the pairwise cosines (N is candidate count or
 * variant count — both kept small), plus O(N·dim) for the Q/A gap.
 */
export function extractRetrievalFeatures(
  topCandidates: ReadonlyArray<{ vector: Float32Array | number[] }> | null,
  queryVector: Float32Array | number[],
  variantVectors: ReadonlyArray<Float32Array | number[]>,
  hypotheticalVectors: ReadonlyArray<Float32Array | number[]>,
): RetrievalFeatures {
  // Duplicate density: pairwise cosine of top candidates.
  const candVecs = (topCandidates ?? []).map(c => c.vector);
  const duplicateDensity = meanPairwiseCosine(candVecs);

  // Intent cohesion: pairwise cosine of variants. Need >=2 to compute.
  const queryIntentCohesion = meanPairwiseCosine(variantVectors);

  // Q/A space gap: distance between question vector and mean of hypotheticals.
  let qaSpaceGap = 0;
  if (hypotheticalVectors.length > 0) {
    const mean = meanVector(hypotheticalVectors);
    qaSpaceGap = 1 - cosineSim(queryVector, mean);
  }

  return {
    duplicateDensity,
    queryIntentCohesion,
    qaSpaceGap,
    candidateCountUsed: candVecs.length,
    variantCount: variantVectors.length,
    hypotheticalCount: hypotheticalVectors.length,
  };
}

/**
 * Apply the routing heuristic to a set of features. Returns the
 * recommended primitive plus the reasoning trace.
 */
export function adaptiveRoute(
  features: RetrievalFeatures,
  options: AdaptiveRouterOptions = {},
): AdaptiveDecision {
  const dupT = options.duplicateThreshold ?? 0.85;
  const intentT = options.intentCohesionThreshold ?? 0.55;
  const gapT = options.qaGapThreshold ?? 0.35;
  const compound = options.preferCompoundWhenMultipleSignals ?? true;

  // Only fire a signal when the underlying input was actually present.
  const mmrSignal = features.candidateCountUsed >= 2 && features.duplicateDensity > dupT;
  const rrfSignal = features.variantCount >= 2 && features.queryIntentCohesion < intentT;
  const hydeSignal = features.hypotheticalCount >= 1 && features.qaSpaceGap > gapT;

  const fireCount = (mmrSignal ? 1 : 0) + (rrfSignal ? 1 : 0) + (hydeSignal ? 1 : 0);

  let primitive: AdaptivePrimitive;
  let reason: string;
  if (fireCount === 0) {
    primitive = 'plain';
    reason = `No signals fired (dup=${features.duplicateDensity.toFixed(3)}<=${dupT}, intentCohesion=${features.queryIntentCohesion.toFixed(3)}>=${intentT}, qaGap=${features.qaSpaceGap.toFixed(3)}<=${gapT}); plain top-k is sufficient.`;
  } else if (fireCount >= 2 && compound) {
    primitive = 'compound';
    const fired = [
      mmrSignal ? `MMR(dup=${features.duplicateDensity.toFixed(3)}>${dupT})` : null,
      rrfSignal ? `RRF(intent=${features.queryIntentCohesion.toFixed(3)}<${intentT})` : null,
      hydeSignal ? `HyDE(qaGap=${features.qaSpaceGap.toFixed(3)}>${gapT})` : null,
    ].filter(Boolean).join(', ');
    reason = `Multiple signals fired: ${fired}; compound primitive composes them all.`;
  } else if (mmrSignal) {
    primitive = 'mmr';
    reason = `MMR signal fired: duplicateDensity=${features.duplicateDensity.toFixed(3)}>${dupT}; corpus has near-duplicate top candidates, diversity rerank wins.`;
  } else if (rrfSignal) {
    primitive = 'rrf';
    reason = `RRF signal fired: queryIntentCohesion=${features.queryIntentCohesion.toFixed(3)}<${intentT}; reformulations cover distinct intents, rank-fusion preserves boundaries.`;
  } else {
    primitive = 'hyde';
    reason = `HyDE signal fired: qaSpaceGap=${features.qaSpaceGap.toFixed(3)}>${gapT}; question and hypothetical answers diverge in vector space, HyDE bridges the gap.`;
  }

  return {
    primitive,
    reason,
    signals: { mmr: mmrSignal, rrf: rrfSignal, hyde: hydeSignal },
    features,
  };
}

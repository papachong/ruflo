#!/usr/bin/env node
/**
 * ADR-121 Phase 16 — Adaptive primitive selection benchmark + witness.
 *
 * Validates the adaptive router on Phase 14's four topology shapes
 * (easy / duplicate-heavy / multi-intent / q-a-gap). For each topology
 * the router examines (corpus top-N density, variant cohesion, Q/A
 * gap) signals and recommends a primitive — we then check that the
 * recommendation matches the topology winner from Phase 14:
 *
 *   easy           → plain (no signals fire)
 *   duplicate-heavy → mmr  (duplicate signal fires)
 *   multi-intent   → rrf  (low intent cohesion fires)
 *   q-a-gap        → hyde (Q/A gap fires)
 *
 * Pure algorithmic benchmark — operates directly on hand-crafted
 * vector topologies (same shapes as Phase 14's topology suite). No
 * mock embeddings; the vectors are intentional test fixtures
 * exercising the routing heuristic.
 *
 * Witness-signed: ed25519 manifest written to bench-witness/.
 *
 * Pass criterion: router picks the correct primitive for each
 * topology (4/4 expected mappings).
 */

import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const embDist = path.join(repoRoot, 'v3/@claude-flow/embeddings/dist');

const { extractRetrievalFeatures, adaptiveRoute } = await import(path.join(embDist, 'adaptive-router.js'));
const { witness, verify, canonicalHash } = await import(path.join(embDist, 'witness.js'));

const argJson = process.argv.includes('--json');
const skipWrite = process.argv.includes('--skip-write');

// =========================================================
// Topology builders — same shapes as Phase 14's topology suite, but
// reshaped so the adaptive router can see them through its inputs:
//   - top-N candidates (for duplicate density)
//   - query reformulation vectors (for intent cohesion)
//   - hypothetical-answer vectors (for Q/A gap)
// =========================================================
const DIM = 16;

function vec(values) {
  const v = new Float32Array(DIM);
  values.forEach((x, i) => { v[i] = x; });
  return v;
}
function unit(values) {
  const v = vec(values);
  let sq = 0;
  for (const x of v) sq += x * x;
  if (sq === 0) return v;
  const n = Math.sqrt(sq);
  const out = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) out[i] = v[i] / n;
  return out;
}

// Topology 1: easy — diverse candidates, cohesive variants, no Q/A gap
function topoEasy() {
  return {
    label: 'easy',
    expectedPrimitive: 'plain',
    topCandidates: [
      { vector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
    ],
    queryVector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    variantVectors: [
      unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0.97, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0.95, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
    hypotheticalVectors: [
      unit([0.95, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0.94, 0, 0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
  };
}

// Topology 2: duplicate-heavy — near-dup candidates, rest is fine
function topoDuplicateHeavy() {
  return {
    label: 'duplicate-heavy',
    expectedPrimitive: 'mmr',
    topCandidates: [
      { vector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.99, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.98, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.97, 0, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.96, 0.01, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
    ],
    queryVector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    variantVectors: [
      unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0.97, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
    hypotheticalVectors: [
      unit([0.95, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
  };
}

// Topology 3: multi-intent — orthogonal variants, candidates moderate
function topoMultiIntent() {
  return {
    label: 'multi-intent',
    expectedPrimitive: 'rrf',
    topCandidates: [
      { vector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.7, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
    ],
    queryVector: unit([0.7, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    variantVectors: [
      unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
    hypotheticalVectors: [
      unit([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
  };
}

// Topology 4: q-a-gap — question in question-space (axis 8), hypotheticals in answer-space (axis 0).
// Top candidates here are the DISTRACTORS that plain search returns (they share the q-space axis 8
// but otherwise spread across different secondary axes — low duplicate density between them).
function topoQAGap() {
  return {
    label: 'q-a-gap',
    expectedPrimitive: 'hyde',
    topCandidates: [
      // All touch axis 8 (q-space) but with distinct secondary signatures so pairwise cosine stays low.
      { vector: unit([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]) },
    ],
    queryVector: unit([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    variantVectors: [
      unit([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
      unit([0, 0, 0, 0, 0, 0, 0, 0, 0.97, 0.03, 0, 0, 0, 0, 0, 0]),
    ],
    hypotheticalVectors: [
      unit([1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
  };
}

// Topology 5: compound (multiple signals fire)
function topoCompound() {
  return {
    label: 'compound-trigger',
    expectedPrimitive: 'compound',
    topCandidates: [
      { vector: unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.99, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: unit([0.98, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) },
    ],
    queryVector: unit([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    variantVectors: [
      unit([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
      unit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
    hypotheticalVectors: [
      unit([1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      unit([1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
  };
}

const topologies = [topoEasy(), topoDuplicateHeavy(), topoMultiIntent(), topoQAGap(), topoCompound()];

if (!argJson) {
  console.log('=== Adaptive primitive router benchmark ===\n');
  console.log(`Topologies: ${topologies.length}`);
  console.log('Pass criterion: router picks expected primitive per topology\n');
}

// =========================================================
// Run
// =========================================================
const results = [];
let correct = 0;
for (const t of topologies) {
  const features = extractRetrievalFeatures(t.topCandidates, t.queryVector, t.variantVectors, t.hypotheticalVectors);
  const decision = adaptiveRoute(features);
  const match = decision.primitive === t.expectedPrimitive;
  if (match) correct++;
  results.push({
    topology: t.label,
    expected: t.expectedPrimitive,
    actual: decision.primitive,
    match,
    features,
    reason: decision.reason,
    signals: decision.signals,
  });
}

const accuracy = correct / topologies.length;

// =========================================================
// Witness
// =========================================================
function getCommit() {
  try { return execSync('git rev-parse HEAD', { cwd: repoRoot, encoding: 'utf8' }).trim(); }
  catch { return null; }
}

const summary = {
  totalTopologies: topologies.length,
  correctPredictions: correct,
  accuracy,
  perTopology: results.map(r => ({
    topology: r.topology,
    expected: r.expected,
    actual: r.actual,
    match: r.match,
    features: {
      duplicateDensity: Number(r.features.duplicateDensity.toFixed(4)),
      queryIntentCohesion: Number(r.features.queryIntentCohesion.toFixed(4)),
      qaSpaceGap: Number(r.features.qaSpaceGap.toFixed(4)),
    },
  })),
};

const manifest = witness({
  benchmark: 'rag-adaptive-router',
  timestamp: new Date().toISOString(),
  commit: getCommit(),
  model: 'algorithmic-test-fixtures-dim16',
  corpus: {
    id: canonicalHash(topologies.map(t => ({ label: t.label, candCount: t.topCandidates.length, variantCount: t.variantVectors.length, hypCount: t.hypotheticalVectors.length }))),
    size: topologies.length,
  },
  queries: { id: canonicalHash(topologies.map(t => ({ label: t.label, expected: t.expectedPrimitive }))), count: topologies.length },
  results: summary,
});

if (!verify(manifest)) {
  console.error('[FAIL] witness self-verify failed');
  process.exit(2);
}

// =========================================================
// Report
// =========================================================
if (argJson) {
  console.log(JSON.stringify({ results, summary, witness: manifest }, null, 2));
} else {
  console.log('### Router decisions vs expected primitive');
  console.log();
  console.log('| topology | expected | actual | match | dup density | intent cohesion | qa gap |');
  console.log('|---|---|---|---:|---:|---:|---:|');
  for (const r of results) {
    const m = r.match ? '✓' : '✗';
    const f = r.features;
    console.log(`| \`${r.topology}\` | ${r.expected} | ${r.actual} | ${m} | ${f.duplicateDensity.toFixed(3)} | ${f.queryIntentCohesion.toFixed(3)} | ${f.qaSpaceGap.toFixed(3)} |`);
  }
  console.log();
  console.log(`Router accuracy: ${correct}/${topologies.length} (${(accuracy * 100).toFixed(0)}%)`);
  console.log();
  console.log('### Routing reasons');
  for (const r of results) {
    console.log(`- \`${r.topology}\` → ${r.actual}: ${r.reason}`);
  }
  console.log();
  console.log('### Witness');
  console.log(`- commit:      ${manifest.commit ?? '(n/a)'}`);
  console.log(`- contentHash: ${manifest.contentHash}`);
  console.log(`- signature:   ${manifest.signature.slice(0, 32)}...`);
  console.log(`- verify():    TRUE`);
}

if (!skipWrite) {
  const witnessDir = path.join(repoRoot, 'bench-witness');
  fs.mkdirSync(witnessDir, { recursive: true });
  const filename = `rag-adaptive-router-${manifest.timestamp.replace(/[:.]/g, '-')}.json`;
  fs.writeFileSync(path.join(witnessDir, filename), JSON.stringify({ results, summary, witness: manifest }, null, 2));
  if (!argJson) console.log(`\nWitness manifest written to bench-witness/${filename}`);
}

// =========================================================
// Pass criterion
// =========================================================
if (correct !== topologies.length) {
  console.error(`[FAIL] router accuracy ${correct}/${topologies.length} below required 100% (no-regression on Phase 14 topology winners)`);
  process.exit(1);
}
process.exit(0);

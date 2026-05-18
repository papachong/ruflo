/**
 * ADR-121 Phase 16 — adaptive router tests.
 *
 * Coverage:
 *  - Feature extraction shape + signal computation
 *  - Each individual signal triggers the right primitive
 *  - Multiple signals → compound (when preferCompound=true)
 *  - No signals → plain
 *  - Custom thresholds change the decision
 *  - Edge cases: insufficient inputs disable the corresponding signal
 */

import { describe, it, expect } from 'vitest';
import {
  extractRetrievalFeatures,
  adaptiveRoute,
} from '../adaptive-router.js';

const DIM = 8;
function vec(values: number[]): Float32Array {
  const v = new Float32Array(DIM);
  values.forEach((x, i) => { v[i] = x; });
  return v;
}

describe('extractRetrievalFeatures', () => {
  it('returns 0 for duplicateDensity when no candidates supplied', () => {
    const f = extractRetrievalFeatures(null, vec([1]), [vec([1]), vec([0, 1])], [vec([1])]);
    expect(f.duplicateDensity).toBe(0);
    expect(f.candidateCountUsed).toBe(0);
  });

  it('high pairwise cosine among candidates → high duplicateDensity', () => {
    const cands = [
      { vector: vec([1, 0]) },
      { vector: vec([0.99, 0.01]) },
      { vector: vec([0.98, 0.02]) },
    ];
    const f = extractRetrievalFeatures(cands, vec([1, 0]), [vec([1, 0])], [vec([1, 0])]);
    expect(f.duplicateDensity).toBeGreaterThan(0.95);
    expect(f.candidateCountUsed).toBe(3);
  });

  it('orthogonal candidates → low duplicateDensity', () => {
    const cands = [
      { vector: vec([1, 0, 0]) },
      { vector: vec([0, 1, 0]) },
      { vector: vec([0, 0, 1]) },
    ];
    const f = extractRetrievalFeatures(cands, vec([1, 0, 0]), [vec([1, 0, 0])], [vec([1, 0, 0])]);
    expect(f.duplicateDensity).toBeLessThan(0.1);
  });

  it('cohesive query variants → high queryIntentCohesion', () => {
    const variants = [vec([1, 0]), vec([0.95, 0.05]), vec([0.9, 0.1])];
    const f = extractRetrievalFeatures(null, vec([1, 0]), variants, [vec([1, 0])]);
    expect(f.queryIntentCohesion).toBeGreaterThan(0.9);
  });

  it('orthogonal query variants → low queryIntentCohesion', () => {
    const variants = [vec([1, 0, 0]), vec([0, 1, 0]), vec([0, 0, 1])];
    const f = extractRetrievalFeatures(null, vec([1, 0, 0]), variants, [vec([1, 0, 0])]);
    expect(f.queryIntentCohesion).toBeCloseTo(0, 5);
  });

  it('single variant → queryIntentCohesion = 0 (not enough to compute)', () => {
    const f = extractRetrievalFeatures(null, vec([1]), [vec([1, 0])], [vec([1, 0])]);
    expect(f.queryIntentCohesion).toBe(0);
  });

  it('question aligned with hypothetical mean → qaSpaceGap near 0', () => {
    const q = vec([1, 0, 0]);
    const hyps = [vec([0.95, 0.05, 0]), vec([0.9, 0.1, 0])];
    const f = extractRetrievalFeatures(null, q, [q], hyps);
    expect(f.qaSpaceGap).toBeLessThan(0.1);
  });

  it('orthogonal question vs hypothetical mean → qaSpaceGap near 1', () => {
    const q = vec([1, 0, 0, 0, 0, 0, 0, 0]);
    const hyps = [vec([0, 0, 0, 0, 1, 0, 0, 0]), vec([0, 0, 0, 0, 0, 1, 0, 0])];
    const f = extractRetrievalFeatures(null, q, [q], hyps);
    expect(f.qaSpaceGap).toBeGreaterThan(0.9);
  });

  it('no hypotheticals → qaSpaceGap = 0', () => {
    const f = extractRetrievalFeatures(null, vec([1]), [vec([1])], []);
    expect(f.qaSpaceGap).toBe(0);
  });
});

describe('adaptiveRoute', () => {
  it('no signals → plain', () => {
    const features = {
      duplicateDensity: 0.3,
      queryIntentCohesion: 0.95,
      qaSpaceGap: 0.05,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('plain');
    expect(d.signals).toEqual({ mmr: false, rrf: false, hyde: false });
  });

  it('high duplicate density alone → mmr', () => {
    const features = {
      duplicateDensity: 0.92,
      queryIntentCohesion: 0.9,
      qaSpaceGap: 0.1,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('mmr');
    expect(d.signals.mmr).toBe(true);
  });

  it('low intent cohesion alone → rrf', () => {
    const features = {
      duplicateDensity: 0.5,
      queryIntentCohesion: 0.3,
      qaSpaceGap: 0.1,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('rrf');
    expect(d.signals.rrf).toBe(true);
  });

  it('large qa gap alone → hyde', () => {
    const features = {
      duplicateDensity: 0.5,
      queryIntentCohesion: 0.9,
      qaSpaceGap: 0.6,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('hyde');
    expect(d.signals.hyde).toBe(true);
  });

  it('multiple signals → compound (default)', () => {
    const features = {
      duplicateDensity: 0.92,
      queryIntentCohesion: 0.3,
      qaSpaceGap: 0.6,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('compound');
    expect(d.signals).toEqual({ mmr: true, rrf: true, hyde: true });
  });

  it('multiple signals + preferCompound=false → picks first matching (mmr > rrf > hyde)', () => {
    const features = {
      duplicateDensity: 0.92,
      queryIntentCohesion: 0.3,
      qaSpaceGap: 0.6,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features, { preferCompoundWhenMultipleSignals: false });
    expect(d.primitive).toBe('mmr');
  });

  it('reason mentions the firing signal', () => {
    const d = adaptiveRoute({
      duplicateDensity: 0.92,
      queryIntentCohesion: 0.9,
      qaSpaceGap: 0.1,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    });
    expect(d.reason).toMatch(/duplicate/i);
  });

  it('custom thresholds change the decision', () => {
    const features = {
      duplicateDensity: 0.7, // below default 0.85
      queryIntentCohesion: 0.95,
      qaSpaceGap: 0.1,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const defaultRoute = adaptiveRoute(features);
    expect(defaultRoute.primitive).toBe('plain');
    const customRoute = adaptiveRoute(features, { duplicateThreshold: 0.6 });
    expect(customRoute.primitive).toBe('mmr');
  });

  it('insufficient candidates disables MMR signal even with high density', () => {
    const features = {
      duplicateDensity: 0.99,
      queryIntentCohesion: 0.9,
      qaSpaceGap: 0.1,
      candidateCountUsed: 0, // disabled
      variantCount: 3,
      hypotheticalCount: 3,
    };
    const d = adaptiveRoute(features);
    expect(d.signals.mmr).toBe(false);
    expect(d.primitive).toBe('plain');
  });

  it('insufficient hypotheticals disables HyDE signal', () => {
    const features = {
      duplicateDensity: 0.3,
      queryIntentCohesion: 0.95,
      qaSpaceGap: 0.99,
      candidateCountUsed: 5,
      variantCount: 3,
      hypotheticalCount: 0, // disabled
    };
    const d = adaptiveRoute(features);
    expect(d.signals.hyde).toBe(false);
    expect(d.primitive).toBe('plain');
  });
});

describe('end-to-end: extract → route', () => {
  it('duplicate-heavy candidates + cohesive variants → mmr', () => {
    const cands = [
      { vector: vec([1, 0, 0, 0, 0, 0, 0, 0]) },
      { vector: vec([0.99, 0.01, 0, 0, 0, 0, 0, 0]) },
      { vector: vec([0.98, 0.02, 0, 0, 0, 0, 0, 0]) },
    ];
    const variants = [vec([1, 0, 0, 0, 0, 0, 0, 0]), vec([0.95, 0.05, 0, 0, 0, 0, 0, 0])];
    const hyps = [vec([0.95, 0.05, 0, 0, 0, 0, 0, 0])];
    const features = extractRetrievalFeatures(cands, vec([1, 0, 0, 0, 0, 0, 0, 0]), variants, hyps);
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('mmr');
  });

  it('multi-intent variants → rrf', () => {
    const cands = [{ vector: vec([1, 0, 0, 0, 0, 0, 0, 0]) }];
    const variants = [
      vec([1, 0, 0, 0, 0, 0, 0, 0]),
      vec([0, 1, 0, 0, 0, 0, 0, 0]),
      vec([0, 0, 1, 0, 0, 0, 0, 0]),
    ];
    const hyps = [vec([0.5, 0.5, 0, 0, 0, 0, 0, 0])];
    const features = extractRetrievalFeatures(cands, vec([1, 0, 0, 0, 0, 0, 0, 0]), variants, hyps);
    const d = adaptiveRoute(features);
    expect(d.primitive).toBe('rrf');
  });
});

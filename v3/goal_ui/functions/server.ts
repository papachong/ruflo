/**
 * Local Hono dev server mounting all `functions/<name>` handlers.
 *
 * `npm run functions:dev` runs this on port 8787.
 * Production replaces this with one-GCF-per-handler deployment.
 *
 * URL shape matches the former Supabase edge functions:
 * `/functions/v1/<name>` — keeps the example.env / GoalInput.tsx
 * URL paths painless across the migration.
 *
 * Security stack (ADR-093 §S2 / Step 22b):
 *   1. CORS allowlist via `RUFLO_ALLOWED_ORIGINS` (comma-separated;
 *      defaults to `localhost:8080,goal.ruv.io`). Same-origin or
 *      header-less callers bypass CORS entirely; cross-origin
 *      browsers see only the allowlisted Access-Control-Allow-Origin.
 *   2. Token check — `X-RuFlo-Token` header MUST equal
 *      `RUFLO_FUNCTIONS_TOKEN`. The server-side default is
 *      `dev-token-change-me` (matches `example.env`'s
 *      `VITE_FUNCTIONS_PUBLIC_TOKEN`) so local dev works out of
 *      the box; production deploys MUST override `RUFLO_FUNCTIONS_TOKEN`.
 *      The token is a WEAK control (it lives in the bundle). The
 *      real defenses are CORS + rate-limit.
 *   3. Per-IP token-bucket rate limit — 60 requests / minute by
 *      default, configurable via `RUFLO_RATE_LIMIT_PER_MIN`. Excess
 *      returns 429.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { serve } from '@hono/node-server';
import { generateResearchGoalHandler } from './generate-research-goal/handler';
import { researchStepHandler } from './research-step/handler';
import { generateActionItemsHandler } from './generate-action-items/handler';
import { optimizeResearchConfigHandler } from './optimize-research-config/handler';

const PORT = Number(process.env.FUNCTIONS_PORT ?? '8787');

const ALLOWED_ORIGINS = (
  process.env.RUFLO_ALLOWED_ORIGINS ?? 'http://localhost:8080,https://goal.ruv.io'
).split(',').map((s) => s.trim()).filter(Boolean);

// Default to the same dev placeholder that `example.env` ships so
// local `npm run functions:dev` works without setup. Production
// deploys MUST set this — Step 22d's CI gate will flag deploys that
// leave it as the default.
const SERVER_TOKEN = process.env.RUFLO_FUNCTIONS_TOKEN ?? 'dev-token-change-me';

const RATE_LIMIT_PER_MIN = Math.max(1, Number(process.env.RUFLO_RATE_LIMIT_PER_MIN ?? '60'));
const RATE_WINDOW_MS = 60_000;

// In-memory token bucket per IP. Acceptable for the LOCAL_FN dev
// server + a single-instance GCF deployment; multi-instance GCF
// would need Redis or Cloud Memorystore (Step 22d concern).
const buckets = new Map<string, { tokens: number; refilledAt: number }>();

function clientIp(c: { req: { header: (n: string) => string | undefined } }): string {
  // Honor X-Forwarded-For from a single hop; collapse multi-hop XFF
  // to the LEFTMOST entry only (origin client). Strip ports.
  const xff = c.req.header('x-forwarded-for');
  if (xff) return xff.split(',')[0]?.trim().split(':')[0] || 'unknown';
  return c.req.header('x-real-ip') || 'unknown';
}

function takeToken(ip: string): boolean {
  const now = Date.now();
  let b = buckets.get(ip);
  if (!b) {
    b = { tokens: RATE_LIMIT_PER_MIN, refilledAt: now };
    buckets.set(ip, b);
  }
  // Refill: linear over the window
  const elapsed = now - b.refilledAt;
  if (elapsed > 0) {
    const refill = (elapsed / RATE_WINDOW_MS) * RATE_LIMIT_PER_MIN;
    b.tokens = Math.min(RATE_LIMIT_PER_MIN, b.tokens + refill);
    b.refilledAt = now;
  }
  if (b.tokens < 1) return false;
  b.tokens -= 1;
  return true;
}

const app = new Hono();

app.use('*', cors({
  origin: (origin) => (ALLOWED_ORIGINS.includes(origin) ? origin : ''),
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'X-RuFlo-Token'],
}));

// Token check (always on). If header missing or wrong → 401.
app.use('/functions/v1/*', async (c, next) => {
  const incoming = c.req.header('X-RuFlo-Token') ?? '';
  if (incoming !== SERVER_TOKEN) {
    return c.json({ error: 'unauthorized' }, 401);
  }
  await next();
});

// Per-IP rate limit. Excess → 429.
app.use('/functions/v1/*', async (c, next) => {
  const ip = clientIp(c);
  if (!takeToken(ip)) {
    return c.json({ error: `rate limit exceeded (${RATE_LIMIT_PER_MIN}/min)` }, 429);
  }
  await next();
});

app.get('/', (c) => c.text(
  'RuFlo functions dev server — endpoints:\n' +
  '  POST /functions/v1/generate-research-goal\n' +
  '  POST /functions/v1/research-step\n' +
  '  POST /functions/v1/generate-action-items\n' +
  '  POST /functions/v1/optimize-research-config\n',
));

app.post('/functions/v1/generate-research-goal', async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const result = await generateResearchGoalHandler({
    category: typeof body?.category === 'string' ? body.category : '',
    customContext: typeof body?.customContext === 'string' ? body.customContext : undefined,
  });
  return c.json(result.body, { status: result.status as 200 | 400 | 402 | 429 | 500 | 502 });
});

app.post('/functions/v1/research-step', async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as Record<string, unknown>;
  const result = await researchStepHandler({
    goal: typeof body.goal === 'string' ? body.goal : '',
    stepTitle: typeof body.stepTitle === 'string' ? body.stepTitle : '',
    stepDescription: typeof body.stepDescription === 'string' ? body.stepDescription : '',
    stepType: typeof body.stepType === 'string' ? body.stepType : '',
    aiModel: typeof body.aiModel === 'string' ? body.aiModel : undefined,
    config: body.config,
    previousStepsData: Array.isArray(body.previousStepsData) ? body.previousStepsData as never : undefined,
  });
  return c.json(result.body, { status: result.status as 200 | 400 | 402 | 429 | 500 | 502 });
});

app.post('/functions/v1/generate-action-items', async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as Record<string, unknown>;
  const result = await generateActionItemsHandler({
    goal: typeof body.goal === 'string' ? body.goal : '',
    researchContext: Array.isArray(body.researchContext) ? body.researchContext as never : [],
    totalSteps: typeof body.totalSteps === 'number' ? body.totalSteps : 0,
    totalDataPoints: typeof body.totalDataPoints === 'number' ? body.totalDataPoints : 0,
  });
  return c.json(result.body, { status: result.status as 200 | 400 | 402 | 429 | 500 | 502 });
});

app.post('/functions/v1/optimize-research-config', async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as { preset?: string; currentGoal?: string };
  const result = await optimizeResearchConfigHandler({
    preset: typeof body.preset === 'string' ? body.preset : '',
    currentGoal: typeof body.currentGoal === 'string' ? body.currentGoal : undefined,
  });
  return c.json(result.body, { status: result.status as 200 | 400 | 402 | 429 | 500 | 502 });
});

serve({ fetch: app.fetch, port: PORT }, (info) => {
  // eslint-disable-next-line no-console
  console.log(`RuFlo functions dev server listening on http://localhost:${info.port}`);
  // eslint-disable-next-line no-console
  console.log(`  POST /functions/v1/generate-research-goal`);
  // eslint-disable-next-line no-console
  console.log(`  Allowed origins: ${ALLOWED_ORIGINS.join(', ')}`);
  // eslint-disable-next-line no-console
  console.log(`  Token: ${SERVER_TOKEN === 'dev-token-change-me' ? 'DEV DEFAULT (set RUFLO_FUNCTIONS_TOKEN before deploy)' : 'configured'}`);
  // eslint-disable-next-line no-console
  console.log(`  Rate limit: ${RATE_LIMIT_PER_MIN} req/min per IP`);
});

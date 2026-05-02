#!/usr/bin/env bash
#
# Smoke test for Step 22b — verifies the LOCAL_FN server enforces:
#   1. 401 when X-RuFlo-Token is missing or wrong
#   2. CORS allowlist (Access-Control-Allow-Origin set only for
#      whitelisted Origins; empty for others)
#   3. 429 when rate limit (default 60 req/min per IP) is exceeded
#
# Expects `npm run functions:dev` (or equivalent) on PORT (default
# 8787). Set RUFLO_RATE_LIMIT_PER_MIN=5 when starting the server
# for the rate-limit test to fire quickly.

set -euo pipefail

PORT="${FUNCTIONS_PORT:-8787}"
URL="http://localhost:${PORT}/functions/v1/generate-research-goal"
GOOD_TOKEN="${RUFLO_FUNCTIONS_TOKEN:-dev-token-change-me}"

pass=0
fail=0

check() {
  local label="$1"
  local cond="$2"
  if [[ "$cond" == "true" ]]; then
    echo "  ✓ $label"
    pass=$((pass + 1))
  else
    echo "  ✘ $label" >&2
    fail=$((fail + 1))
  fi
}

echo "Server: $URL"
echo ""

# 1. No token → 401
echo "[1/4] No X-RuFlo-Token → 401"
status=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$URL" -H "Content-Type: application/json" -d '{"category":"finance"}')
check "got HTTP $status (expected 401)" "$([[ "$status" == "401" ]] && echo true || echo false)"

# 2. Wrong token → 401
echo "[2/4] Wrong X-RuFlo-Token → 401"
status=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$URL" -H "Content-Type: application/json" -H "X-RuFlo-Token: WRONG" -d '{"category":"finance"}')
check "got HTTP $status (expected 401)" "$([[ "$status" == "401" ]] && echo true || echo false)"

# 3. CORS allowlist: disallowed Origin gets empty Access-Control-Allow-Origin (browser blocks)
echo "[3/4] Disallowed Origin → no allow-origin header"
hdr=$(curl -sS -i -X OPTIONS "$URL" -H "Origin: https://evil.example.com" -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: content-type,x-ruflo-token" 2>&1 | tr -d '\r' | grep -i '^access-control-allow-origin:' || true)
allow_origin_value="${hdr#*: }"
check "no allowlisted origin echoed back (got '$allow_origin_value')" "$([[ -z "$allow_origin_value" || "$allow_origin_value" == "" ]] && echo true || echo false)"

# 4. Rate limit (run with RUFLO_RATE_LIMIT_PER_MIN=5 to make this fast).
# Send 12 requests; expect at least one 429.
echo "[4/4] Burst beyond rate limit → 429"
hits429=0
for i in $(seq 1 12); do
  s=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$URL" -H "Content-Type: application/json" -H "X-RuFlo-Token: $GOOD_TOKEN" -d '{"category":"finance"}')
  [[ "$s" == "429" ]] && hits429=$((hits429 + 1))
done
check "saw at least one 429 in 12 requests (got $hits429)" "$([[ "$hits429" -ge 1 ]] && echo true || echo false)"

echo ""
echo "Passed: $pass  Failed: $fail"
exit $fail

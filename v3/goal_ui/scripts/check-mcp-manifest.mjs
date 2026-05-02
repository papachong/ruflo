#!/usr/bin/env tsx
/**
 * R-5.1 manifest validator.
 *
 * Validates `functions/mcp/mcp-server.json` against the shape the
 * MCP TypeScript SDK expects for tools/list responses (the
 * `Tool` interface). Each tool entry must have:
 *
 *   - name: non-empty string
 *   - description?: string
 *   - inputSchema: { type: "object", properties: {...}, required?: [...] }
 *
 * The server-level fields (name, version, description, transport,
 * command, args) match the convention used by `claude mcp add`
 * configs and the platform's existing MCP servers.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

let pass = 0, fail = 0;
function check(label, ok, detail) {
  if (ok) { console.log(`  ✓ ${label}`); pass++; }
  else { console.log(`  ✘ ${label}${detail ? ` — ${detail}` : ''}`); fail++; }
}

const path = resolve('functions/mcp/mcp-server.json');
const raw = readFileSync(path, 'utf8');
let manifest;
try { manifest = JSON.parse(raw); }
catch (e) {
  console.error('manifest is not valid JSON:', e.message);
  process.exit(1);
}

console.log('R-5.1 MCP manifest validation\n');

// Server-level fields
check('manifest is object', typeof manifest === 'object' && manifest !== null);
check('manifest.name is non-empty string', typeof manifest.name === 'string' && manifest.name.length > 0);
check('manifest.version matches semver-ish', typeof manifest.version === 'string' && /^\d+\.\d+\.\d+/.test(manifest.version));
check('manifest.description is string', typeof manifest.description === 'string' && manifest.description.length > 0);
check('manifest.transport is "stdio"', manifest.transport === 'stdio');
check('manifest.command is string', typeof manifest.command === 'string');
check('manifest.args is array of strings', Array.isArray(manifest.args) && manifest.args.every(a => typeof a === 'string'));

// Tools array
check('manifest.tools is array', Array.isArray(manifest.tools));
check('exactly 5 tools', manifest.tools.length === 5);

const expectedToolNames = [
  'generate_research_goal',
  'research_step',
  'generate_action_items',
  'optimize_research_config',
  'run_full_research',
];
for (const expected of expectedToolNames) {
  const found = manifest.tools.find((t) => t.name === expected);
  check(`tool "${expected}" present`, !!found);
}

// Per-tool MCP Tool shape validation
for (const tool of manifest.tools) {
  console.log(`\nValidating tool: ${tool.name}`);
  check(`  name is non-empty string`, typeof tool.name === 'string' && tool.name.length > 0);
  check(`  name matches /^[a-z][a-z0-9_]*$/`, /^[a-z][a-z0-9_]*$/.test(tool.name));
  check(`  description is non-empty string`, typeof tool.description === 'string' && tool.description.length > 0);
  check(`  inputSchema is object`, typeof tool.inputSchema === 'object' && tool.inputSchema !== null);
  check(`  inputSchema.type === 'object'`, tool.inputSchema?.type === 'object');
  check(`  inputSchema.properties is object`, typeof tool.inputSchema?.properties === 'object' && tool.inputSchema.properties !== null);
  check(`  inputSchema.properties is non-empty`, Object.keys(tool.inputSchema.properties ?? {}).length > 0);
  if (tool.inputSchema.required !== undefined) {
    check(`  inputSchema.required is array of strings`,
      Array.isArray(tool.inputSchema.required) &&
      tool.inputSchema.required.every((s) => typeof s === 'string'));
    // every required field must be in properties
    const propKeys = new Set(Object.keys(tool.inputSchema.properties));
    const missing = tool.inputSchema.required.filter((r) => !propKeys.has(r));
    check(`  every required field exists in properties`, missing.length === 0,
      missing.length ? `missing: ${missing.join(', ')}` : '');
  }

  // Each property must have a `type` (or `oneOf`/`enum`/`$ref`)
  for (const [propName, propSchema] of Object.entries(tool.inputSchema.properties)) {
    const hasType = typeof propSchema?.type === 'string'
      || Array.isArray(propSchema?.type)
      || propSchema?.oneOf
      || propSchema?.enum
      || propSchema?.$ref;
    check(`  property "${propName}" has type/oneOf/enum`, hasType);
  }
}

console.log(`\nPassed: ${pass}  Failed: ${fail}`);
process.exit(fail);

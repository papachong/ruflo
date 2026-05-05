#!/usr/bin/env node
/**
 * @claude-flow/cli-core entry point.
 *
 * Status: alpha (ADR-100). Foundation surface exported here; full memory +
 * hooks command implementations land in subsequent alphas.
 *
 * Stable exports today (Step 2 of ADR-100 Plan of work):
 *   - types: CommandContext, Command, CommandResult, ParsedFlags, ...
 *   - output: terminal printing + tables + spinners + progress
 *   - fs-secure: path-traversal guards + atomic write
 *   - mcp-tools/types: MCPTool, MCPToolInputSchema, MCPToolResult
 *   - mcp-tools/validate-input: input bounds and shape validators
 *
 * Coming in Step 3+:
 *   - mcp-tools/memory: memory_* tool definitions
 *   - mcp-tools/hooks: hooks_* tool definitions
 *   - commands/memory: memory store/list/retrieve/search/delete/init/...
 *   - commands/hooks: hooks route/post-edit/post-task/model-outcome/...
 */

import { fileURLToPath } from 'node:url';

// Re-export foundation surface so plugin authors can pin to cli-core.
export * from './types.js';
export * as output from './output.js';
export type { MCPTool, MCPToolInputSchema, MCPToolResult } from './mcp-tools/types.js';
export * as validateInput from './mcp-tools/validate-input.js';

// Bin entry — runs when invoked as `claude-flow-core <command>`.
if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('cli-core/dist/src/index.js')) {
  const args = process.argv.slice(2);

  if (args[0] === '--version' || args[0] === '-v') {
    const url = new URL('../../package.json', import.meta.url);
    const fs = await import('node:fs/promises');
    const pkg = JSON.parse(await fs.readFile(fileURLToPath(url), 'utf-8'));
    console.log(pkg.version);
    process.exit(0);
  }

  if (args[0] === '--help' || args[0] === '-h' || args.length === 0) {
    console.log(`@claude-flow/cli-core (alpha — ADR-100 Step 2)

Lite core surface of @claude-flow/cli — memory + hooks only. Designed to load
fast on cold npx cache (<5s) so plugin skills don't race the 30s MCP-startup
timeout (#1748 Issue 3).

Currently published surface (programmatic):
  import { CommandContext, output, MCPTool } from '@claude-flow/cli-core';
  import { ... } from '@claude-flow/cli-core/output';
  import { ... } from '@claude-flow/cli-core/types';
  import { ... } from '@claude-flow/cli-core/mcp-tools/memory';   (Step 3)
  import { ... } from '@claude-flow/cli-core/mcp-tools/hooks';    (Step 3)

CLI surface (memory + hooks subcommands) lands in alpha.1 — for now use:
  npx @claude-flow/cli@alpha <command>

Track progress: https://github.com/ruvnet/ruflo/issues/1760`);
    process.exit(0);
  }

  console.error(`@claude-flow/cli-core: command "${args[0]}" not yet wired into the alpha skeleton.
For now, use the full CLI:  npx @claude-flow/cli@alpha ${args.join(' ')}
`);
  process.exit(1);
}

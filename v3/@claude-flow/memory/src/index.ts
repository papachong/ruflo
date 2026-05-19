/**
 * @claude-flow/memory - V3 Unified Memory System
 *
 * Provides a unified memory interface backed by AgentDB with HNSW indexing
 * for 150x-12,500x faster vector search compared to brute-force approaches.
 *
 * @module @claude-flow/memory
 *
 * @example
 * ```typescript
 * import { UnifiedMemoryService, query, QueryTemplates } from '@claude-flow/memory';
 *
 * // Initialize the memory service
 * const memory = new UnifiedMemoryService({
 *   dimensions: 1536,
 *   cacheEnabled: true,
 *   embeddingGenerator: async (text) => embeddings.embed(text),
 * });
 *
 * await memory.initialize();
 *
 * // Store entries
 * await memory.store({
 *   key: 'auth-patterns',
 *   content: 'OAuth 2.0 implementation patterns for secure authentication',
 *   tags: ['auth', 'security', 'patterns'],
 * });
 *
 * // Semantic search
 * const results = await memory.semanticSearch('user authentication best practices', 5);
 *
 * // Query with fluent builder
 * const entries = await memory.query(
 *   query()
 *     .semantic('security vulnerabilities')
 *     .inNamespace('security')
 *     .withTags(['critical'])
 *     .threshold(0.8)
 *     .limit(10)
 *     .build()
 * );
 * ```
 */

// ===== Core Types =====
export type {
  // Memory Entry Types
  MemoryType,
  AccessLevel,
  ConsistencyLevel,
  DistanceMetric,
  MemoryEntry,
  MemoryEntryInput,
  MemoryEntryUpdate,

  // Query Types
  QueryType,
  MemoryQuery,
  SearchResult,
  SearchOptions,

  // HNSW Types
  HNSWConfig,
  HNSWStats,
  QuantizationConfig,

  // Backend Types
  IMemoryBackend,
  BackendStats,
  HealthCheckResult,
  ComponentHealth,

  // Cache Types
  CacheConfig,
  CacheStats,
  CachedEntry,

  // Migration Types
  MigrationSource,
  MigrationConfig,
  MigrationProgress,
  MigrationResult,
  MigrationError,

  // Event Types
  MemoryEventType,
  MemoryEvent,
  MemoryEventHandler,

  // SONA Types
  SONAMode,
  LearningPattern,

  // Utility Types
  EmbeddingGenerator,
} from './types.js';

// Utility Functions and Constants (runtime values)
export {
  generateMemoryId,
  createDefaultEntry,
  PERFORMANCE_TARGETS,
} from './types.js';

// ===== Auto Memory Bridge (ADR-048) =====
export { AutoMemoryBridge, resolveAutoMemoryDir, findGitRoot } from './auto-memory-bridge.js';
export type {
  AutoMemoryBridgeConfig,
  MemoryInsight,
  InsightCategory,
  SyncDirection,
  SyncMode,
  PruneStrategy,
  SyncResult,
  ImportResult,
} from './auto-memory-bridge.js';

// ===== Learning Bridge =====
export { LearningBridge } from './learning-bridge.js';
export type {
  LearningBridgeConfig,
  LearningStats,
  ConsolidateResult,
  PatternMatch,
} from './learning-bridge.js';

// ===== RVF Learning Persistence (ADR-057 Phase 6) =====
export { RvfLearningStore } from './rvf-learning-store.js';
export type {
  RvfLearningStoreConfig,
  PatternRecord,
  LoraRecord,
  EwcRecord,
  TrajectoryRecord,
} from './rvf-learning-store.js';
export { PersistentSonaCoordinator } from './persistent-sona.js';
export type { PersistentSonaConfig } from './persistent-sona.js';

// ===== RVF Migration (Bidirectional) =====
export { RvfMigrator } from './rvf-migration.js';
export type { RvfMigrationOptions, RvfMigrationResult } from './rvf-migration.js';

// ===== Knowledge Graph =====
export { MemoryGraph } from './memory-graph.js';
export type {
  MemoryGraphConfig,
  GraphNode,
  GraphEdge,
  GraphStats,
  RankedResult,
  EdgeType,
} from './memory-graph.js';

// ===== Agent-Scoped Memory =====
export {
  resolveAgentMemoryDir,
  createAgentBridge,
  transferKnowledge,
  listAgentScopes,
} from './agent-memory-scope.js';
export type {
  AgentMemoryScope,
  AgentScopedConfig,
  TransferOptions,
  TransferResult,
} from './agent-memory-scope.js';

// ===== Controller Registry (ADR-053) =====
export { ControllerRegistry, INIT_LEVELS } from './controller-registry.js';
export type {
  AgentDBControllerName,
  CLIControllerName,
  ControllerName,
  InitLevel,
  ControllerHealth,
  RegistryHealthReport,
  RuntimeConfig,
} from './controller-registry.js';

// ===== Core Components =====
export { AgentDBAdapter } from './agentdb-adapter.js';
export type { AgentDBAdapterConfig } from './agentdb-adapter.js';
export { AgentDBBackend } from './agentdb-backend.js';
export type { AgentDBBackendConfig } from './agentdb-backend.js';
export { SQLiteBackend } from './sqlite-backend.js';
export type { SQLiteBackendConfig } from './sqlite-backend.js';
export { SqlJsBackend } from './sqljs-backend.js';
export type { SqlJsBackendConfig } from './sqljs-backend.js';
export { HybridBackend } from './hybrid-backend.js';
export type {
  HybridBackendConfig,
  StructuredQuery,
  SemanticQuery,
  HybridQuery,
} from './hybrid-backend.js';
// `RvfBackend` and `HnswLite` are intentionally NOT re-exported from the top level
// per ADR-125 Phase 1. They remain internal — RvfBackend is reachable via
// `createDatabase({ provider: 'rvf' })`, and HnswLite is used inside RvfBackend.
// Importers that need them directly should import the explicit module path.
export { cosineSimilarity } from './hnsw-lite.js';
export { HNSWIndex } from './hnsw-index.js';
export { CacheManager, TieredCacheManager } from './cache-manager.js';
export { QueryBuilder, query, QueryTemplates } from './query-builder.js';
export type { SortDirection, SortField } from './query-builder.js';
export { MemoryMigrator, createMigrator, migrateMultipleSources } from './migration.js';
export { createDatabase, getPlatformInfo, getAvailableProviders } from './database-provider.js';
export type { DatabaseProvider, DatabaseOptions } from './database-provider.js';

// ===== Smart Retrieval (ADR-090) =====
export { smartSearch, defaultQueryExpansions } from './smart-retrieval.js';
export type {
  SearchCandidate,
  RawSearchRequest,
  RawSearchResponse,
  SearchFn,
  SmartSearchOptions,
  SmartSearchStats,
  SmartSearchResult,
} from './smart-retrieval.js';

// ===== Unified Memory Service =====
import { EventEmitter } from 'node:events';
import {
  IMemoryBackend,
  MemoryEntry,
  MemoryEntryInput,
  MemoryEntryUpdate,
  MemoryQuery,
  SearchResult,
  SearchOptions,
  BackendStats,
  HealthCheckResult,
  EmbeddingGenerator,
  MigrationSource,
  MigrationConfig,
  MigrationResult,
} from './types.js';
import { AgentDBAdapter, AgentDBAdapterConfig } from './agentdb-adapter.js';
import { MemoryMigrator } from './migration.js';

/**
 * Configuration for UnifiedMemoryService
 */
export interface UnifiedMemoryServiceConfig extends Partial<AgentDBAdapterConfig> {
  /** Enable automatic embedding generation */
  autoEmbed?: boolean;

  /** Default embedding dimensions */
  dimensions?: number;

  /** Embedding generator function */
  embeddingGenerator?: EmbeddingGenerator;
}

/**
 * Memory Service implementation (legacy class name).
 *
 * @deprecated Use {@link MemoryService} — the canonical name introduced by
 *   ADR-125 Phase 1. `UnifiedMemoryService` is preserved as an alias and will
 *   continue to work through `@claude-flow/memory@3.0.0-rc`. Both names refer
 *   to the same class.
 *
 * High-level interface that provides:
 * - Simple API for common operations
 * - Automatic embedding generation
 * - Cross-agent memory sharing
 * - SONA integration for learning
 * - Event-driven notifications
 * - Performance monitoring
 *
 * @see {@link MemoryService} for the canonical alias.
 */
export class UnifiedMemoryService extends EventEmitter implements IMemoryBackend {
  private adapter: AgentDBAdapter;
  private config: UnifiedMemoryServiceConfig;
  private initialized: boolean = false;

  /**
   * The active memory backend. Defaults to the `AgentDBAdapter` created from
   * config, but can be any `IMemoryBackend` implementation (e.g. `HybridBackend`
   * when constructed via `createHybridService` per ADR-009 / ADR-125 Phase 2).
   *
   * Public so consumers can introspect the backend type without reaching for
   * `getAdapter()` (which is AgentDB-specific).
   */
  public backend: IMemoryBackend;

  constructor(config: UnifiedMemoryServiceConfig = {}) {
    super();
    this.config = {
      dimensions: 1536,
      cacheEnabled: true,
      autoEmbed: true,
      ...config,
    };

    this.adapter = new AgentDBAdapter({
      dimensions: this.config.dimensions,
      cacheEnabled: this.config.cacheEnabled,
      cacheSize: this.config.cacheSize,
      cacheTtl: this.config.cacheTtl,
      hnswM: this.config.hnswM,
      hnswEfConstruction: this.config.hnswEfConstruction,
      defaultNamespace: this.config.defaultNamespace,
      embeddingGenerator: this.config.embeddingGenerator,
      persistenceEnabled: this.config.persistenceEnabled,
      persistencePath: this.config.persistencePath,
      maxEntries: this.config.maxEntries,
    });

    // Default backend is the AgentDB adapter — ADR-125 Phase 2 introduces the
    // ability to replace it via `withBackend()` / `createHybridService`.
    this.backend = this.adapter;

    // Forward adapter events
    this.adapter.on('entry:stored', (data) => this.emit('entry:stored', data));
    this.adapter.on('entry:updated', (data) => this.emit('entry:updated', data));
    this.adapter.on('entry:deleted', (data) => this.emit('entry:deleted', data));
    this.adapter.on('cache:hit', (data) => this.emit('cache:hit', data));
    this.adapter.on('cache:miss', (data) => this.emit('cache:miss', data));
    this.adapter.on('index:added', (data) => this.emit('index:added', data));
  }

  /**
   * Replace the active backend with a pre-built `IMemoryBackend`.
   *
   * Used by `createHybridService` (ADR-125 Phase 2) to wire `HybridBackend`
   * through `createDatabase` rather than instantiating `AgentDBAdapter`
   * directly. The legacy `AgentDBAdapter` instance is kept around for the
   * `storeEntry` / `semanticSearch` convenience methods that the IMemoryBackend
   * interface doesn't cover; those calls still flow through it.
   *
   * Returns `this` for chaining.
   *
   * @internal Prefer the factory functions (`createHybridService`,
   *   `createPersistentService`, etc.) over calling this directly.
   */
  withBackend(backend: IMemoryBackend): this {
    this.backend = backend;
    return this;
  }

  // ===== Lifecycle =====

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.backend.initialize();
    // If the active backend is something other than the adapter (e.g. a
    // HybridBackend wired by createHybridService), the adapter may never be
    // used — skip its initialize() in that case to avoid double-allocating.
    this.initialized = true;
    this.emit('initialized');
  }

  async shutdown(): Promise<void> {
    if (!this.initialized) return;
    await this.backend.shutdown();
    this.initialized = false;
    this.emit('shutdown');
  }

  // ===== IMemoryBackend Implementation =====

  async store(entry: MemoryEntry): Promise<void> {
    return this.backend.store(entry);
  }

  async get(id: string): Promise<MemoryEntry | null> {
    return this.backend.get(id);
  }

  async getByKey(namespace: string, key: string): Promise<MemoryEntry | null> {
    return this.backend.getByKey(namespace, key);
  }

  async update(id: string, update: MemoryEntryUpdate): Promise<MemoryEntry | null> {
    return this.backend.update(id, update);
  }

  async delete(id: string): Promise<boolean> {
    return this.backend.delete(id);
  }

  async query(query: MemoryQuery): Promise<MemoryEntry[]> {
    return this.backend.query(query);
  }

  async search(embedding: Float32Array, options: SearchOptions): Promise<SearchResult[]> {
    return this.backend.search(embedding, options);
  }

  async bulkInsert(entries: MemoryEntry[]): Promise<void> {
    return this.backend.bulkInsert(entries);
  }

  async bulkDelete(ids: string[]): Promise<number> {
    return this.backend.bulkDelete(ids);
  }

  async count(namespace?: string): Promise<number> {
    return this.backend.count(namespace);
  }

  async listNamespaces(): Promise<string[]> {
    return this.backend.listNamespaces();
  }

  async clearNamespace(namespace: string): Promise<number> {
    return this.backend.clearNamespace(namespace);
  }

  async getStats(): Promise<BackendStats> {
    return this.backend.getStats();
  }

  async healthCheck(): Promise<HealthCheckResult> {
    return this.backend.healthCheck();
  }

  // ===== Convenience Methods =====

  /**
   * Store an entry from simple input.
   *
   * When the active backend is the default AgentDBAdapter, delegates to its
   * native `storeEntry`. When a custom backend is wired (e.g. HybridBackend
   * via `createHybridService`), this builds a full `MemoryEntry` and stores
   * it through the IMemoryBackend interface.
   */
  async storeEntry(input: MemoryEntryInput): Promise<MemoryEntry> {
    if (this.backend === this.adapter) {
      return this.adapter.storeEntry(input);
    }
    // Generic path for non-AgentDBAdapter backends
    const { createDefaultEntry } = await import('./types.js');
    const entry = createDefaultEntry(input as any);
    // Generate an embedding on demand if needed
    if (!entry.embedding && this.config.embeddingGenerator && entry.content) {
      try {
        entry.embedding = await this.config.embeddingGenerator(entry.content);
      } catch {
        // Leave embedding undefined; backend may still accept the entry.
      }
    }
    await this.backend.store(entry);
    return entry;
  }

  /**
   * Semantic search by content string.
   *
   * When the active backend is the default AgentDBAdapter, delegates to its
   * native `semanticSearch`. Otherwise generates an embedding via the
   * configured generator and calls `backend.search()`.
   */
  async semanticSearch(
    content: string,
    k: number = 10,
    threshold?: number
  ): Promise<SearchResult[]> {
    if (this.backend === this.adapter) {
      return this.adapter.semanticSearch(content, k, threshold);
    }
    if (!this.config.embeddingGenerator) {
      throw new Error(
        'semanticSearch requires an embeddingGenerator when backend is not the AgentDBAdapter'
      );
    }
    const embedding = await this.config.embeddingGenerator(content);
    return this.backend.search(embedding, { k, threshold });
  }

  /**
   * Find similar entries to a given entry
   */
  async findSimilar(id: string, k: number = 5): Promise<SearchResult[]> {
    const entry = await this.get(id);
    if (!entry || !entry.embedding) {
      return [];
    }

    const results = await this.search(entry.embedding, { k: k + 1 });

    // Filter out the source entry
    return results.filter((r) => r.entry.id !== id).slice(0, k);
  }

  /**
   * Get or create an entry
   */
  async getOrCreate(
    namespace: string,
    key: string,
    creator: () => MemoryEntryInput | Promise<MemoryEntryInput>
  ): Promise<MemoryEntry> {
    const existing = await this.getByKey(namespace, key);
    if (existing) return existing;

    const input = await creator();
    return this.storeEntry({ ...input, namespace, key });
  }

  /**
   * Append content to an existing entry
   */
  async appendContent(id: string, content: string): Promise<MemoryEntry | null> {
    const entry = await this.get(id);
    if (!entry) return null;

    return this.update(id, {
      content: entry.content + '\n' + content,
    });
  }

  /**
   * Add tags to an existing entry
   */
  async addTags(id: string, tags: string[]): Promise<MemoryEntry | null> {
    const entry = await this.get(id);
    if (!entry) return null;

    const newTags = [...new Set([...entry.tags, ...tags])];
    return this.update(id, { tags: newTags });
  }

  /**
   * Remove tags from an existing entry
   */
  async removeTags(id: string, tags: string[]): Promise<MemoryEntry | null> {
    const entry = await this.get(id);
    if (!entry) return null;

    const newTags = entry.tags.filter((t) => !tags.includes(t));
    return this.update(id, { tags: newTags });
  }

  // ===== Migration =====

  /**
   * Migrate from a legacy memory source.
   *
   * The migrator is AgentDB-specific (writes through `AgentDBAdapter`).
   * When a custom backend is wired (e.g. HybridBackend), migration still
   * targets the local AgentDB adapter; the hybrid backend can pick up the
   * migrated entries on next read via its own AgentDB index.
   */
  async migrateFrom(
    source: MigrationSource,
    sourcePath: string,
    options: Partial<MigrationConfig> = {}
  ): Promise<MigrationResult> {
    const migrator = new MemoryMigrator(
      this.adapter,
      { source, sourcePath, ...options },
      this.config.embeddingGenerator
    );

    // Forward migration events
    migrator.on('migration:started', (data) => this.emit('migration:started', data));
    migrator.on('migration:progress', (data) => this.emit('migration:progress', data));
    migrator.on('migration:completed', (data) => this.emit('migration:completed', data));
    migrator.on('migration:failed', (data) => this.emit('migration:failed', data));
    migrator.on('migration:error', (data) => this.emit('migration:error', data));
    migrator.on('migration:warning', (data) => this.emit('migration:warning', data));

    return migrator.migrate();
  }

  // ===== Cross-Agent Memory Sharing =====

  /**
   * Share an entry with another agent
   */
  async shareWith(id: string, agentId: string): Promise<MemoryEntry | null> {
    const entry = await this.get(id);
    if (!entry) return null;

    const sharedWith = (entry.metadata.sharedWith as string[]) || [];
    if (!sharedWith.includes(agentId)) {
      sharedWith.push(agentId);
    }

    return this.update(id, {
      metadata: { ...entry.metadata, sharedWith },
    });
  }

  /**
   * Get entries shared with a specific agent
   */
  async getSharedWith(agentId: string): Promise<MemoryEntry[]> {
    const all = await this.query({ type: 'hybrid', limit: 10000 });
    return all.filter((entry) => {
      const sharedWith = (entry.metadata.sharedWith as string[]) || [];
      return sharedWith.includes(agentId);
    });
  }

  // ===== Utility =====

  /**
   * Get the underlying adapter for advanced operations
   */
  getAdapter(): AgentDBAdapter {
    return this.adapter;
  }

  /**
   * Check if the service is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}

// ===== Canonical Alias (ADR-125 Phase 1) =====

/**
 * Canonical memory service entry point (ADR-125).
 *
 * `MemoryService` is the preferred name as of `@claude-flow/memory@3.0.0-alpha.18`.
 * It is an alias of {@link UnifiedMemoryService}; both names refer to the same
 * class so existing callers continue working unchanged.
 *
 * @example
 * ```typescript
 * import { MemoryService } from '@claude-flow/memory';
 *
 * const memory = new MemoryService({ dimensions: 1536 });
 * await memory.initialize();
 * ```
 */
export const MemoryService = UnifiedMemoryService;

/**
 * @public
 * @typedef MemoryService
 *
 * Type alias matching the canonical {@link MemoryService} runtime export so that
 * `import type { MemoryService } from '@claude-flow/memory'` works alongside the
 * value import.
 */
export type MemoryService = UnifiedMemoryService;

/**
 * Config type alias for {@link MemoryService}.
 */
export type MemoryServiceConfig = UnifiedMemoryServiceConfig;

// ===== Factory Functions =====

/**
 * Create a simple in-memory service (for testing)
 */
export function createInMemoryService(): UnifiedMemoryService {
  return new UnifiedMemoryService({
    persistenceEnabled: false,
    cacheEnabled: true,
  });
}

/**
 * Create a persistent memory service
 */
export function createPersistentService(path: string): UnifiedMemoryService {
  return new UnifiedMemoryService({
    persistenceEnabled: true,
    persistencePath: path,
    cacheEnabled: true,
  });
}

/**
 * Create a memory service with embedding support
 */
export function createEmbeddingService(
  embeddingGenerator: EmbeddingGenerator,
  dimensions: number = 1536
): UnifiedMemoryService {
  return new UnifiedMemoryService({
    embeddingGenerator,
    dimensions,
    autoEmbed: true,
    cacheEnabled: true,
  });
}

/**
 * Create a hybrid memory service (SQLite + AgentDB).
 *
 * This is the DEFAULT recommended configuration per ADR-009. ADR-125 Phase 2
 * delivers the real wiring: the returned service's backend is a `HybridBackend`
 * created through `createDatabase({ provider: 'hybrid' })`, not an AgentDB-only
 * downgrade as in earlier versions.
 *
 * @example
 * ```typescript
 * const memory = await createHybridService('./data/memory.db', embeddingFn);
 * await memory.initialize();
 *
 * // Structured queries go to SQLite
 * const user = await memory.getByKey('users', 'john@example.com');
 *
 * // Semantic queries go to AgentDB
 * const similar = await memory.semanticSearch('authentication patterns', 10);
 *
 * // Verify the backend is actually hybrid
 * import { HybridBackend } from '@claude-flow/memory';
 * memory.backend instanceof HybridBackend; // true
 * ```
 */
export async function createHybridService(
  databasePath: string,
  embeddingGenerator: EmbeddingGenerator,
  dimensions: number = 1536
): Promise<UnifiedMemoryService> {
  const { createDatabase } = await import('./database-provider.js');
  const hybridBackend = await createDatabase(databasePath, {
    provider: 'hybrid',
    embeddingGenerator,
    dimensions,
  });
  const service = new UnifiedMemoryService({
    embeddingGenerator,
    dimensions,
    autoEmbed: true,
    cacheEnabled: true,
    persistenceEnabled: true,
    persistencePath: databasePath,
  });
  return service.withBackend(hybridBackend);
}

// Default export
export default UnifiedMemoryService;

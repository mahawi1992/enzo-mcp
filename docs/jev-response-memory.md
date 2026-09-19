# Jev response memory

Enzo's optional response memory is a local, bounded SQLite ledger inspired by the
fingerprint-and-answer model described by [JevCache](https://jevcache.sh/#share).
It is deliberately an Enzo-owned boundary rather than an automatic connection to the
public JevCache index.

## Invariants

1. **Approval precedes memory.** Enzo rebuilds and validates the approved dispatch
   manifest before it asks the ledger for a response. A preview or digest mismatch
   performs no cache read and no provider call.
2. **Exact means exact.** The cache key binds Enzo's canonical manifest digest to a
   caller-selected namespace, explicit model epoch, and cache schema version.
3. **Requests are not retained.** The database stores the manifest digest and Jev's
   response JSON, never selected context, evidence, investigation identity, atom
   identity, or approval references.
4. **Cached answers remain untrusted input.** Enzo verifies the record hash and parses
   the typed Jev response again. It then applies the current local answer contract and
   decision thresholds. Replay evidence carries the cache creation time, expiry time,
   and response hash instead of presenting an old answer as newly observed.
5. **Failure is soft.** An unavailable or corrupt ledger behaves like a cache miss.
   Provider behavior and deterministic-evidence precedence do not change.
6. **Reuse is bounded.** Every entry has a TTL and the ledger has a maximum number of
   entries. Oldest-accessed entries are removed first.
7. **Local files are private.** Enzo hardens existing and new cache directories to
   owner-only access and database/WAL/SHM files to owner-only read/write access. It
   rejects symbolic-link cache files and sidecars.

## Configuration

The cache is disabled unless `ENZO_JEV_CACHE_DIR` is present. When enabled, both
`ENZO_JEV_CACHE_NAMESPACE` and `ENZO_JEV_CACHE_MODEL_EPOCH` are required.

| Variable | Meaning | Default |
| --- | --- | --- |
| `ENZO_JEV_CACHE_DIR` | Private directory containing `jev-responses.sqlite3` | disabled |
| `ENZO_JEV_CACHE_NAMESPACE` | Deployment/trust-boundary scope | required |
| `ENZO_JEV_CACHE_MODEL_EPOCH` | Immutable effective-model version | required |
| `ENZO_JEV_CACHE_TTL_SECONDS` | Record lifetime, 1 second to 30 days | `86400` |
| `ENZO_JEV_CACHE_MAX_ENTRIES` | Record limit, 1 to 100,000 | `10000` |

Rotate the model epoch whenever model behavior, prompts, or provider semantics change.
Namespaces should distinguish projects and environments that must not share responses.

## Why public sharing is off

The [JevCache distribution repository](https://github.com/hyperspaceai/jevcache)
currently ships a prebuilt v0.1 sidecar and documents a global read/write index. Enzo
cannot yet independently inspect the server implementation or rely on documented TTL,
deletion, invalidation, and tenancy guarantees. Its documented redaction-based cache
identity also has different equivalence semantics from Enzo's exact approved request.

Consequently, Enzo does not download, execute, or publish to that service automatically.
A future adapter can implement the existing response-cache interface after its contract
and threat boundary are reviewable. That preserves the useful memory abstraction without
weakening Enzo's consent or data-minimization guarantees.

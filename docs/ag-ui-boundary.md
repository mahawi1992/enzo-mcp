# AG-UI integration boundary

AG-UI can present an Enzo outbound manifest, interrupt a run for review, and carry
the typed decision into a resumed run. It does not prove that a person approved the
request, persist an investigation across process failure, or guarantee replay of a
dropped event stream.

The smallest safe integration keeps these responsibilities separate:

```text
Enzo builds canonical Jev manifest
              |
              v
AG-UI state exposes the exact manifest
              |
              v
AG-UI interrupt requests approve or reject
              |
              v
host resumes with the approved SHA-256 digest
              |
              v
Enzo rebuilds and compares before any Jev call
```

## Required invariants

1. The review UI displays the canonical logical request and digest returned by
   Enzo. It must not reconstruct the payload independently.
2. A resume decision includes the reviewed digest. A plain Boolean approval is not
   sufficient.
3. Enzo rebuilds the request from current state and sends only when its digest still
   matches. Any selected-data mutation requires another review.
4. AG-UI snapshots and deltas are presentation state. A durable store remains
   responsible for pending proposals, investigation state, and restart recovery.
5. The host records who or what asserted approval when that information exists.
   Enzo treats the approval reference as correlation metadata, not proof of human
   consent.
6. An interrupted or disconnected run is safe by default: no matching resume means
   no external call.

## Suggested event mapping

| Enzo event | AG-UI representation |
| --- | --- |
| Manifest prepared | Shared-state snapshot or delta containing the manifest |
| Review required | Tool-bound interrupt with approve and reject choices |
| Review completed | Resume input containing decision and manifest digest |
| Jev dispatched | Tool result carrying the same manifest and sensor result |
| Manifest changed | New state update and a new interrupt; prior approval is invalid |

This adapter belongs above Enzo's three MCP tools. It should not add a fourth MCP
tool or move orchestration into Enzo.

## Durability follow-up

If restart recovery becomes a requirement, add a versioned internal store with
compare-and-swap semantics for investigations and pending manifests. Store the
canonical logical request and digest before emitting the interrupt. AG-UI can then
re-expose that stored state after restart, but the store—not AG-UI—is what makes the
recovery durable.

References:

- [AG-UI state management](https://github.com/ag-ui-protocol/ag-ui/blob/main/docs/concepts/state.mdx)
- [AG-UI events](https://github.com/ag-ui-protocol/ag-ui/blob/main/docs/concepts/events.mdx)
- [AG-UI resumable transport discussion](https://github.com/ag-ui-protocol/ag-ui/issues/2105)

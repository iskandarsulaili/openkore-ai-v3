# Macro publication and hot-reload workflow

## Scope

This document covers the **implemented** macro publication and reload flow shared across:

- the Python sidecar compiler and publisher
- the generated artifact files in [`control`](../../control)
- the OpenKore bridge reload action
- the existing `macro` and `eventMacro` plugin surfaces

## Files involved

### Source-side request contract

- `POST /v1/macros/publish`
- request contract in [`ai_sidecar/contracts/macros.py`](../ai_sidecar/contracts/macros.py)

### Generated artifacts

| File | Current role |
| --- | --- |
| [`control/ai_sidecar_generated_macros.txt`](../../control/ai_sidecar_generated_macros.txt) | Target file for the `macro` plugin |
| [`control/ai_sidecar_generated_eventmacros.txt`](../../control/ai_sidecar_generated_eventmacros.txt) | Target file for the `eventMacro` plugin |
| [`control/ai_sidecar_macros.txt`](../../control/ai_sidecar_macros.txt) | Stable catalog pointer file |
| [`control/ai_sidecar_macro_manifest.json`](../../control/ai_sidecar_macro_manifest.json) | Latest publication metadata |

## End-to-end workflow

1. A caller sends `MacroPublishRequest`.
2. The sidecar compiles normalized macro text and event macro text.
3. The sidecar writes the generated files atomically into `control/`.
4. The sidecar persists publication metadata.
5. The sidecar optionally enqueues a `macro_reload` action for the target bot.
6. The bridge polls `/v1/actions/next`.
7. The bridge executes the internal safe macro reload flow.
8. The bridge acknowledges success or failure through `/v1/acknowledgements/action`.

## Compiler rules implemented now

The compiler lives in [`ai_sidecar/domain/macro_compiler.py`](../ai_sidecar/domain/macro_compiler.py).

### Naming rules

- routine and automacro names must match `^[A-Za-z0-9_.:-]+$`
- invalid names raise an exception and fail publication

### Line normalization

- carriage returns and newlines inside submitted lines are replaced with spaces
- leading and trailing whitespace is trimmed

### Duplicate-name resolution

Within a single publish request:

- duplicate macro names are collapsed by name
- duplicate event macro names are collapsed by name
- duplicate automacro names are collapsed by name
- the **last occurrence wins** before final name-sorted output

### Ordering

After normalization, emitted items are sorted by name.

### Empty automacro condition handling

If an automacro is submitted without any usable conditions, the compiler inserts:

```text
BaseLevel >= 1
```

This is an implemented compatibility safeguard.

## Publication metadata

The compiler derives:

- `content_sha256`
- `publication_id` from the first 24 hex chars of the digest
- `version` from UTC timestamp plus digest prefix

The persisted manifest records:

- file names
- counts
- macro names
- event macro names
- automacro names

## Artifact writing and rollback

The publisher snapshots existing contents of all target files before writing.

If any write fails:

- previous contents are restored where available
- newly created files are removed if they had no previous content
- the publication request fails

This rollback covers file publication only.

## Reload action queueing

If `enqueue_reload = true`, the sidecar creates an action with:

- `kind = macro_reload`
- `priority_tier = macro_management`
- `conflict_key = reload_conflict_key` (default `macro_reload`)
- `idempotency_key = macro-reload:{content_sha256}`

Operational consequence:

- duplicate reloads for the same artifact content are deduplicated by idempotency key
- only one queued reload with the same conflict key is intended to survive queue arbitration

## Bridge-side reload execution

The bridge handles `kind = macro_reload` specially.

It validates metadata with narrow allow rules:

- macro file names must be plain filenames, no path separators
- plugin names must match a restricted identifier regex

Then it executes these OpenKore commands internally:

```text
conf macro_file <macro_file>
plugin reload <macro_plugin>
conf eventMacro_file <event_macro_file>
plugin reload <event_macro_plugin>
```

## Relationship to command policy

[`control/ai_sidecar_policy.txt`](../../control/ai_sidecar_policy.txt) denies external `plugin` and `conf` command roots.

That policy applies to normal `kind = command` actions.

The internal `macro_reload` flow is different:

- it does **not** pass through the normal command allow/deny filter
- it runs through the bridge's dedicated safe reload path

This distinction is deliberate in the current implementation.

## Existing plugin surfaces used by reload

### `macro` plugin

[`plugins/macro/macro.pl`](../../plugins/macro/macro.pl) reloads its control file when `macro_file` changes and on `plugin reload macro`.

### `eventMacro` plugin

[`plugins/eventMacro/eventMacro.pl`](../../plugins/eventMacro/eventMacro.pl) reloads its control file when `eventMacro_file` changes and on `plugin reload eventMacro`.

## Failure semantics

### Publish request failures

Examples:

- invalid names
- file write failures
- repository persistence failure during the publication path

Result:

- API returns `published = false`
- no reload action is queued

### Reload execution failures

Examples:

- bridge config disables macro reload
- safe filename/plugin validation falls back or fails a step
- OpenKore command execution throws an error

Result:

- action acknowledgement returns failure
- the published files remain on disk
- there is no automatic restoration of the previous in-client macro configuration

## Operational notes

- published artifact paths in API responses are relative to the workspace root
- the catalog file is a stable machine-readable bridge between operators and the generated artifacts
- the manifest file is the best source for the latest publication id, version, counts, and names

## Implemented now vs extension points

### Implemented now

- macro compilation, artifact publication, manifesting, queueing, and bridge-driven hot reload
- rollback for file publication failures
- safe internal reload orchestration through existing OpenKore command pathways

### Future extension points, not implemented now

- transactional rollback of successful file publication after failed in-client reload
- richer static validation of macro and eventMacro source semantics before publish
- per-bot isolated artifact directories instead of the current shared generated filenames

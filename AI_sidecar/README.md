# OpenKore AI Sidecar (local)

Local FastAPI sidecar for OpenKore bridge IPC.

## Quick start

1. Create environment file:
   - copy `.env.example` to `.env`
2. Install:
   - `python -m pip install -e .`
3. Run:
   - `openkore-ai-sidecar`

Default bind: `127.0.0.1:18081`.

## Implemented endpoints

- `GET /v1/health/live`
- `GET /v1/health/ready`
- `POST /v1/ingest/register`
- `POST /v1/ingest/snapshot`
- `POST /v1/telemetry/ingest`
- `POST /v1/actions/queue`
- `POST /v1/actions/next`
- `POST /v1/acknowledgements/action`

## Runtime behavior

- In-memory bot registry
- In-memory latest snapshot cache per bot
- In-memory action queue per bot with:
  - idempotency dedupe (`idempotency_key`)
  - conflict-key exclusion (`conflict_key`)
  - expiry handling
  - dispatch + acknowledgement state transitions
- Latency budget tracking with explicit no-op fallback response for action polling


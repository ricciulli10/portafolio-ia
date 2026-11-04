# Security

No credentials live in this repository. Everything sensitive comes from environment
variables, and `.env` is git-ignored (see `.env.example`).

| Control | Where |
| --- | --- |
| Optional API key, compared in constant time | `src/credit_risk/api.py` |
| Strict input validation with size limits, unknown fields rejected | `src/credit_risk/api.py` |
| Error messages reveal no paths or stack traces | `src/credit_risk/api.py`, `src/agentkit/runtime.py` |
| Container runs as a non-root user | `Dockerfile` |
| Webhooks: HMAC signature, replay window, idempotency, fails closed | `src/intake_bot/app.py` |
| Published ports bound to localhost; n8n image pinned to a current release | `docker-compose.yml` |
| Secret scanning and static analysis on every push | `.github/workflows/ci.yml` |
| Tool input validated; sensitive actions denied unless a human approves | `src/agentkit/runtime.py` |
| Customer text treated as data; decisions gated by code, not by the prompt | `src/underwriter/domain.py` |

The model file is a joblib pickle. Only load artifacts you trained yourself, because
unpickling an untrusted file can run code.

This is a portfolio project on public data. Add rate limiting, TLS and real
authentication before exposing it to the internet.

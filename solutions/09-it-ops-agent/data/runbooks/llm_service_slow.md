# Runbook: LLM service slow or not responding (sample)

**Symptoms:** answers take more than 30 seconds, `/health` times out, users report "the assistant is stuck".

1. Check the service: `systemctl status llama-server@granite-8b` (tool `service_status`).
2. Check the health endpoint: `http://127.0.0.1:8080/health` (tool `check_http`).
3. Check CPU and memory: diagnostics `load`, `top_cpu`, `memory`. llama-server should use about one thread per
   physical core; if load is far above the core count, another job (fine-tuning, re-indexing) is competing.
4. Check SMT: diagnostic `smt` should show `SMT=4` for inference LPARs.
5. Check the LPAR entitlement: diagnostic `lpar_info` (shared vs dedicated processors, capped mode).
6. If the service is down or hung, propose a restart (needs approval). Do not restart during business hours
   without informing the service owner.
7. If slowness persists after restart: open a ticket (severity **medium**) with the diagnostics attached.

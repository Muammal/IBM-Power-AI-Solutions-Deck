# Runbook: filesystem full (sample)

**Symptoms:** applications fail to write, `df -h` shows a filesystem at 90% or more, services crash with
"No space left on device".

1. Find the filesystem: `df -hT` (diagnostic `disk`).
2. Find the biggest directories: `du -xh --max-depth=2 /var | sort -h | tail` (run manually, as an operator).
3. Common causes on AI LPARs: Hugging Face cache (`/var/lib/ai/hf-cache`), old GGUF files in `/opt/models`,
   `journalctl` logs, `out/` checkpoints from fine-tuning.
4. Safe clean-up: `journalctl --vacuum-size=500M`; delete old checkpoints only after confirming the promoted model.
5. If the filesystem is a PowerVS volume, extend it: `ibmcloud pi volume action` / on-prem: ask the storage team.
6. Open a ticket with severity **high** if a production filesystem is above 95%.

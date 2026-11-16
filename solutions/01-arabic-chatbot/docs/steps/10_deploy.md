# Step 10 — Run as a service

## Goal

Keep the chatbot running after you log out and after a reboot. There are two options; pick **one**:

| | A. systemd + venv | B. Podman container |
|---|---|---|
| Uses | the `venv` from Step 03 | an image built from `deploy/Containerfile` |
| Good for | one LPAR, simplest to debug | repeatable deployments, several LPARs, OpenShift later |
| llama.cpp backend | yes (built in Step 09) | transformers only as shipped (extend the image to build llama-cpp-python) |

## Files created in this step

| File | Purpose |
|------|---------|
| `deploy/chatbot.service` | systemd unit: runs `venv/bin/python app.py` as user `chatbot`, hardened, offline Hub |
| `deploy/Containerfile` | UBI 9 Python 3.12 image (ppc64le); installs `requirements.txt` from the IBM index |
| `docs/steps/10_deploy.md` | This document |

## Commands to run (copy-paste ready)

### Option A — systemd + venv

```bash
# 1. A dedicated system user with no login shell
sudo useradd --system --home-dir /var/lib/chatbot --shell /sbin/nologin chatbot
sudo mkdir -p /var/lib/chatbot && sudo chown chatbot:chatbot /var/lib/chatbot

# 2. The service reads .env (HF_TOKEN), so let only root and the chatbot group read it
sudo chown root:chatbot /opt/power-chatbot/.env && sudo chmod 640 /opt/power-chatbot/.env

# 3. Pre-download the model into the service's cache (the service itself runs offline)
MODEL=$(/opt/power-chatbot/venv/bin/python -c "import yaml;print(yaml.safe_load(open('/opt/power-chatbot/config.yaml'))['model']['id'])")
sudo -u chatbot bash -c "set -a; . /opt/power-chatbot/.env; set +a; \
  HF_HOME=/var/lib/chatbot/hf-cache /opt/power-chatbot/venv/bin/hf download $MODEL"

# 4. Install and start
sudo cp /opt/power-chatbot/deploy/chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chatbot
journalctl -u chatbot -f          # watch it load; Ctrl+C to stop watching
```

If Step 08 showed that pinning to one NUMA node is faster, pin the service the same way. `llm.py` then counts only
that node's cores:

```bash
sudo systemctl edit chatbot      # add these 3 lines in the editor that opens:
#   [Service]
#   ExecStart=
#   ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 /opt/power-chatbot/venv/bin/python app.py
```

With `backend: llamacpp`, put the GGUF where the service can read it (e.g. `/opt/power-chatbot/models/`) and skip step 3.

### Option B — Podman container

Build **on the LPAR** (the image must be ppc64le):

```bash
cd /opt/power-chatbot
sudo dnf install -y podman
podman build -f deploy/Containerfile -t power-chatbot .

# A named volume keeps downloaded models across container restarts
podman volume create chatbot-hf

# First start: downloads the model into the volume (needs HF_TOKEN for Llama)
podman run -d --name chatbot \
  -p 127.0.0.1:7860:7860 \
  --env-file .env \
  -v chatbot-hf:/opt/app-root/src/.cache/huggingface:Z \
  -v ./config.yaml:/opt/app-root/src/config.yaml:ro,Z \
  power-chatbot
podman logs -f chatbot
```

Mounting `config.yaml` read-only means you can **swap models without rebuilding**: edit the file, then
`podman restart chatbot`.

To start the container at boot, use a Quadlet file (Podman ≥ 4.4, RHEL 9.2+):

```bash
sudo tee /etc/containers/systemd/chatbot.container >/dev/null <<'EOF'
[Container]
Image=localhost/power-chatbot
PublishPort=127.0.0.1:7860:7860
EnvironmentFile=/opt/power-chatbot/.env
Environment=HF_HUB_OFFLINE=1
Volume=chatbot-hf:/opt/app-root/src/.cache/huggingface:Z
Volume=/opt/power-chatbot/config.yaml:/opt/app-root/src/config.yaml:ro,Z
[Service]
Restart=on-failure
TimeoutStartSec=900
[Install]
WantedBy=multi-user.target
EOF
podman rm -f chatbot            # remove the manual one first
sudo systemctl daemon-reload && sudo systemctl start chatbot
```

(Rootful Quadlet uses root's volume `chatbot-hf`. Run the download once with `sudo podman run ...` as above.)

## Expected output

```
$ systemctl status chatbot
● chatbot.service - Power chatbot (FastAPI web UI + LLM on CPU)
     Loaded: loaded (/etc/systemd/system/chatbot.service; enabled; preset: disabled)
     Active: active (running) since ...
$ journalctl -u chatbot | tail -3
Ready: transformers | meta-llama/Llama-3.2-3B-Instruct | bfloat16 | 16 threads
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:7860 (Press CTRL+C to quit)
```

## How to verify

```bash
curl -s http://127.0.0.1:7860/health
sudo systemctl restart chatbot && sleep 60 && curl -s http://127.0.0.1:7860/health   # survives a restart
```

**Passes when** `/health` returns `{"status":"ok",...,"busy":false}` both times, and the browser test from
Step 07 (via the SSH tunnel) works. Optionally reboot the LPAR and check again.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `status=203/EXEC` | Wrong `ExecStart` path or venv not readable by `chatbot` | `ls -l /opt/power-chatbot/venv/bin/python`; `chmod -R o+rX /opt/power-chatbot` |
| `LocalEntryNotFoundError` / "offline mode" | Model not in `/var/lib/chatbot/hf-cache` | Repeat step 3 of Option A (same `HF_HOME`) |
| `PermissionError: /opt/power-chatbot/.env` | Group permissions | `chown root:chatbot .env; chmod 640 .env` |
| Service restarts in a loop, `Killed` in the journal | Out of memory | Smaller model / bf16 / GGUF; check `journalctl -k | grep -i oom` |
| `exec format error` in Podman | Image built on x86 | Build on the ppc64le LPAR |
| Container is `unhealthy` for the first minutes | Model still loading | `--start-period=10m` already allows for this; check `podman logs` |
| Browser can't reach it | Published on 127.0.0.1 only | SSH tunnel (Step 07), or a reverse proxy with TLS; avoid exposing port 7860 directly |

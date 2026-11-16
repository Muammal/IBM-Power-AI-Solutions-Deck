# Step 07 — Web UI

## Goal

Serve a browser chat page that streams answers token by token and renders Arabic **right-to-left**
and English left-to-right, automatically, in the same conversation.

### Why FastAPI and not Gradio?

The brief said: use Gradio if it installs cleanly on ppc64le, otherwise FastAPI with one HTML page.
While building this template we checked the IBM wheel snapshot `linux-v2026.03.31`, the index the
ppc64le/pyeco examples use. **It has no Gradio build**, while FastAPI + uvicorn are already pinned and
validated in pyeco's `vllm-example`. Gradio would also pull in ~25 extra packages (pandas, pillow, orjson, ...)
for a page we can write in ~140 lines of HTML. So this template uses **FastAPI + one HTML page**.

## Files created in this step

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server: `GET /` page, `GET /health`, `POST /chat` (streaming) |
| `static/index.html` | The chat page: plain HTML/CSS/JS, no framework, `dir="auto"` everywhere |
| `docs/steps/07_web_ui.md` | This document |

## How it works

```
browser (index.html)                         app.py (FastAPI)                     llm.py
  history = [...]  -- POST /chat {messages} -->  validate (pydantic)
                                                 asyncio.Lock: one answer at a time
                    <-- text/plain chunks ----  iterate_in_threadpool(llm.generate) --> model
  TextDecoder(stream) -> bubble.textContent
```

* **Stateless server.** The conversation lives in the browser (`history`). Each request sends it, and
  `llm.build_messages` trims it and adds the system prompt. Restarting the server loses nothing important.
* **Validation.** `role` must be `user` or `assistant`, so a browser cannot inject its own `system` prompt.
  Messages are capped at 8,000 characters and 100 turns.
* **One answer at a time.** A CPU has a fixed number of cores, so two parallel answers would each run at about half speed.
  An `asyncio.Lock` makes requests queue. `/health` shows `"busy": true` while one is running.
* **Streaming.** `llm.generate()` yields text pieces. `iterate_in_threadpool` runs each blocking step in a worker
  thread, so the server stays responsive. The browser reads the body with `response.body.getReader()`.
  `TextDecoder(..., {stream: true})` keeps an Arabic character intact when its two UTF-8 bytes arrive in
  different chunks.
* **Stop / closed tab.** Between pieces the server checks `request.is_disconnected()`. When the user clicks
  **Stop** or closes the tab, it sets the `stop` event, the model halts at its next token, and the lock is freed.
* **Right-to-left.** Each bubble and the input box have `dir="auto"`, so the browser picks the direction from the
  first strong letter. `unicode-bidi: plaintext` applies that per line, so a mixed answer still
  renders well. Text is inserted with `textContent` (never `innerHTML`), so model output cannot inject HTML.
* **Safe default address.** `server.host: 127.0.0.1` means only the LPAR itself can connect. Reach it through an
  SSH tunnel (below), or put it behind your organisation's reverse proxy with TLS and login.

## Commands to run (copy-paste ready)

On the LPAR:

```bash
cd /opt/power-chatbot && source venv/bin/activate
python app.py
```

On your laptop, open a tunnel in a second terminal, then browse to http://localhost:7860:

```bash
ssh -N -L 7860:127.0.0.1:7860 <user>@<LPAR_IP>
```

To listen on all interfaces instead (only on a trusted network, and open the firewall):

```bash
CHATBOT_HOST=0.0.0.0 python app.py
sudo firewall-cmd --add-port=7860/tcp
```

## Expected output

Server terminal:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
Loading model ...
Ready: transformers | meta-llama/Llama-3.2-3B-Instruct | bfloat16 | 32 threads
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:7860 (Press CTRL+C to quit)
```

Browser: a chat page whose header shows the model line from `/health`.

## How to verify

1. **Health:** `curl -s http://127.0.0.1:7860/health` → `{"status":"ok","model":"transformers | ...","busy":false}`
2. **Streaming from the command line:** `-N` turns off curl's buffering, so the text should appear word by word:
   ```bash
   curl -N -s -X POST http://127.0.0.1:7860/chat -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"اكتب ثلاث جمل عن مدينة بغداد."}]}'
   ```
3. **In the browser:** ask `اكتب ثلاث جمل عن مدينة بغداد.` and check that:
   * your question and the answer are **right-aligned and read right-to-left**, and list numbers sit on the right;
   * the answer **grows while you watch** (streaming), and the **Stop** button appears while it runs;
   * then ask an English question: that bubble is left-to-right.
4. **Stop works:** start a long answer, click **Stop**, then check that `/health` shows `"busy": false` within a second or two.

**Passes when all four checks succeed.**

> Tested while building this template, on x86 with the pinned packages and `Qwen/Qwen2.5-0.5B-Instruct`. In the browser,
> the Arabic bubbles had computed `direction: rtl` and the English ones `ltr`. The answer text grew in 39 visible steps
> between the first text (3.5 s) and the end (10 s). A `system` role from the client was rejected with 422, and a
> history not ending in a user turn with 400. The first version kept generating for ~15 s after a client hung up; the
> `stop`-event design above fixes that.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Application startup failed` with `ConnectionError ... huggingface.co` | transformers asks the Hub for metadata on **every** load, even when the model is cached | After the first download, run with `HF_HUB_OFFLINE=1` (put it in `.env`) |
| Browser cannot connect | Server listens on 127.0.0.1 only | Use the SSH tunnel, or `CHATBOT_HOST=0.0.0.0` + `firewall-cmd` |
| Whole answer appears at once | A proxy buffers the response | The app sends `X-Accel-Buffering: no`. Disable buffering on your proxy too |
| Second user waits | By design: one answer at a time | Add LPARs/replicas behind a load balancer for more users |
| Arabic shows LTR in the page | Old cached page | Hard-refresh (Ctrl+F5) |
| `422 Unprocessable Entity` | Request body doesn't match the schema | Send `{"messages":[{"role":"user","content":"..."}]}` |
| Port already in use | Another app.py is running | `ss -ltnp | grep 7860`, stop it, or set `server.port` |

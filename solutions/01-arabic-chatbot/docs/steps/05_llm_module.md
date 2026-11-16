# Step 05 — Model layer (`llm.py`)

## Goal

Build one small module that hides *which* model and *which* engine are used. The CLI, the web UI
and the benchmark all call the same functions, so swapping a model never touches them.

```python
import llm
cfg = llm.load_config()                       # reads config.yaml (+ .env)
llm.load_model(cfg)                           # downloads on first use, then loads into RAM
for piece in llm.generate([{"role": "user", "content": "مرحبا"}]):   # stream=True by default
    print(piece, end="", flush=True)
```

## Files created in this step

| File | Purpose |
|------|---------|
| `llm.py` | `load_config`, `load_model(cfg)`, `generate(messages, stream=True)`, plus small helpers |
| `docs/steps/05_llm_module.md` | This document: a line-by-line walkthrough |

## Walkthrough for beginners

### Imports and module state (lines 1–11)

* `gc, os, re, threading`: garbage collection (free RAM), environment variables, regular expressions, threads.
* `yaml`: reads `config.yaml`. `load_dotenv` reads `.env`. `TemplateError` is the error Jinja2 raises
  when a chat template rejects a conversation.
* `_ARABIC`: a regular expression that matches one character from the Arabic Unicode blocks.
* `_state = {}`: a plain dictionary holding whatever `load_model` loaded. Using a module-level dict keeps
  the API to simple functions, with no classes for beginners to learn.

### `load_config(path=None)`

1. `load_dotenv()` copies `KEY=value` lines from `.env` into the process environment, so `HF_TOKEN` becomes visible.
2. The config path is the argument, else the `CHATBOT_CONFIG` environment variable, else `config.yaml`.
3. `yaml.safe_load` turns the YAML into nested Python dicts. `safe_load` never executes code from the file.

### `detect_language(text)`

1. Keep only letters (`c.isalpha()`). Digits, spaces and punctuation don't say anything about language.
2. Count how many are Arabic script.
3. If at least 30% are Arabic, return `"ar"`, else `"en"`. Iraqi dialect (`شلونك؟`) is Arabic script, so it is detected as `ar`.
   Messages that mix in English technical terms ("شنو هو Kubernetes؟") still count as Arabic.

### `physical_cores()`: a Power-specific detail

PyTorch chooses its default thread count with a helper library (`cpuinfo`) that does not support POWER, so on an
LPAR it starts **one thread per hardware thread**: 32 on 8 cores in SMT4, 64 in SMT8. For LLM inference that usually
means too many threads fighting over each core's MMA units and memory bandwidth.

1. `os.sched_getaffinity(0)` lists the hardware threads this process may run on (it respects `taskset`/`numactl`).
2. For each one, Linux's `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` names all threads of the same
   core (for example `0-3` in SMT4). Threads of one core share the same text.
3. The number of distinct texts is the number of physical cores.
4. On a system without these files (a Windows or macOS laptop), it returns `0`: "let the library decide".

### `load_model(cfg)`

1. `unload()` frees any previous model first, so loading a second model does not need double the RAM.
2. Read `backend` and `runtime.threads`. `0` means "one thread per physical core" (`physical_cores()`).
3. **transformers branch**
   * `OMP_NUM_THREADS` is set **before** `import torch`. PyTorch's matrix multiplies go to the OpenBLAS library bundled in
     IBM's wheel (Step 03), and OpenBLAS and the OpenMP runtime read this variable once, for every thread they create,
     including the background thread that runs `generate()`.
   * Imports happen *inside* the branch, so the llama.cpp backend works without loading PyTorch.
   * Only `float32` and `bfloat16` are allowed. `float16` is slow and less accurate on CPUs.
   * `torch.set_num_threads(threads)` also sets the thread count. It matters when a process loads a model a second time
     with another value (the benchmark does that), because the environment variable is only read once.
   * `kw` holds the shared arguments: `revision` (which version of the repo), `token` (for gated models)
     and `trust_remote_code` (keep `false`: it would run Python code downloaded from the model repo).
   * `AutoTokenizer.from_pretrained` downloads (first time) and loads the tokenizer. It turns text into token IDs and back.
   * `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)` loads the weights. `getattr(torch, "bfloat16")`
     turns the string from YAML into the real `torch.bfloat16` type. `.eval()` switches off training-only behaviour.
   * If the tokenizer has no `chat_template`, it is a base model, not a chat model, and we stop with a clear message.
4. **llamacpp branch**: `Llama(model_path=...)` loads a GGUF file. `n_ctx` is the context window, `n_batch` the
   prompt batch size, and `n_threads` is the same thread count as above (`None` = llama.cpp's default). No `chat_format` is passed, so llama.cpp
   uses the chat template stored **inside** the GGUF file, which is the model's own template.
5. Remember `cfg` and `backend` in `_state`.

### `build_messages(history)`

1. If the caller already put a `system` message first, use it unchanged.
2. Keep only the last `max_history_turns` turns. On a CPU, every extra token of history makes the
   **time to first token** longer, because the whole prompt is processed again for each answer.
   `2 * turns - 1` = the newest user message plus `turns - 1` earlier user/assistant pairs.
3. Drop leading assistant messages. Some templates (ALLaM, Llama-2 style) raise an error unless the
   conversation starts with `user` and alternates strictly.
4. Pick the system prompt in the language of the newest user message and put it first.

### `_without_system_role(messages)`

Some chat templates (for example Gemma's) have no `system` role and raise `TemplateError`. In that case we
merge the system text into the first user message and try again. The prompt format still comes from the model's own template.

### `_stream_transformers(messages)`: the core

1. `tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)`
   * applies **the model's own chat template** (a Jinja2 template shipped in `tokenizer_config.json`).
     Llama wraps turns in `<|start_header_id|>…<|eot_id|>`, Qwen uses `<|im_start|>…<|im_end|>`, ALLaM uses
     `[INST]…[/INST]`. We never write those strings ourselves, which is why model swaps need no code changes.
   * `add_generation_prompt=True` appends the "assistant, your turn" marker.
   * `return_dict=True` returns `input_ids` **and** `attention_mask`, ready for `generate(**inputs)`.
2. `stop = threading.Event()` plus the tiny `StopOnEvent` class form a **stopping criterion**: on every new token,
   `generate` asks "should I stop?". We say yes once `stop` is set, which happens when the reader goes away.
3. `TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)` is a queue. `generate` pushes tokens
   into it, and our `for` loop reads decoded text out of it. `skip_prompt` hides the echoed prompt.
4. `kwargs` collects the generation settings from `config.yaml`. `pad_token_id` falls back to the EOS token
   (Llama has no pad token, which would otherwise produce a warning).
5. Temperature: `> 0` means sampling (`do_sample=True`, with `temperature` and `top_p`). `0` means greedy decoding. Then we also
   clear `temperature/top_p/top_k` defaults that the model's `generation_config.json` may set, to avoid warnings.
6. `run()` calls `model.generate(**kwargs)` in a **background thread**, because `generate` blocks until
   the answer is complete. If it fails (for example out of memory), it saves the error and ends the streamer,
   so the reader is not left waiting forever.
7. `yield from (text for text in streamer if text)` hands each piece of text to the caller as soon as it exists.
   Empty strings are skipped. The streamer emits `""` while it waits for a complete word, which happens often with Arabic.
8. `finally: stop.set(); thread.join()` runs even if the caller stops early (Ctrl+C, closed browser tab).
   Generation stops at the next token and the thread ends cleanly.
9. If the thread recorded an error, raise it in the caller's thread.

### `_stream_llamacpp(messages)`

`create_chat_completion(..., stream=True)` is llama.cpp's OpenAI-style chat API. Each chunk looks like
`{"choices": [{"delta": {"content": "..."}}]}`. We yield only the text. Setting names map 1:1
(`repetition_penalty` → `repeat_penalty`).

### `generate(messages, stream=True)`

1. Add the system prompt and trim the history (`build_messages`).
2. Choose the backend's streaming function. Both produce the same thing: an iterator of text pieces.
3. `stream=True` returns the iterator. `stream=False` joins it into one string.

Because both backends share this shape, `cli.py` and `app.py` do not know or care which backend is active.

### `count_tokens`, `describe`, `unload`

* `count_tokens(text)` uses the active backend's tokenizer. The benchmark uses it to compute tokens per second.
* `describe()` returns a one-line summary such as `transformers | Qwen/... | bfloat16 | 32 threads`.
* `unload()` empties `_state` and runs the garbage collector, so the old model's RAM is released.

## Commands to run (copy-paste ready)

```bash
cd /opt/power-chatbot && source venv/bin/activate
python - <<'EOF'
import llm, time
cfg = llm.load_config()
t = time.time(); llm.load_model(cfg); print("loaded in %.1fs:" % (time.time() - t), llm.describe())
print("EN ->", llm.generate([{"role": "user", "content": "What is the capital of France? One sentence."}], stream=False))
print("AR -> ", end="")
for piece in llm.generate([{"role": "user", "content": "ما هي عاصمة العراق؟ أجب بجملة قصيرة."}]):
    print(piece, end="", flush=True)
print()
EOF
```

The first run downloads the model (6–15 GB) into `~/.cache/huggingface`. Later runs load from disk.

## Expected output

```
loaded in 9.8s: transformers | meta-llama/Llama-3.2-3B-Instruct | bfloat16 | 32 threads
EN -> The capital of France is Paris.
AR -> عاصمة العراق هي بغداد.
```

> Tested while building this template, on x86 with the same pinned versions (torch 2.10.0, transformers 4.57.6) and the
> small stand-in model `Qwen/Qwen2.5-0.5B-Instruct`. Result: `EN -> The capital of France is Paris.`; the streamed Arabic
> answer contained `بغداد`, and the Arabic system prompt was selected for the Iraqi input `شلونك؟`.

## How to verify

**Passes** when the English line mentions Paris and the Arabic line, printed piece by piece, contains بغداد.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `has no chat template. Use an -Instruct/chat model.` | `model.id` points to a base model | Use the `-Instruct` repo |
| `TemplateError: Conversation roles must alternate` | History starting with `assistant` or two `user` turns in a row | `build_messages` already fixes the start. Make sure your UI appends the assistant reply before the next user turn |
| `Killed` / `MemoryError` while loading | Not enough RAM | Use `dtype: bfloat16`, a 3B model, or the GGUF backend |
| Loading takes minutes every time | Model is being downloaded again (`HF_HOME` differs between users) | Run as one user, or set `HF_HOME` to a shared path |
| `ValueError: model.dtype must be float32 or bfloat16` | `float16` in config | Use `bfloat16` |
| `ModuleNotFoundError: llama_cpp` | `backend: llamacpp` before Step 09 | Run Step 09 or switch back to `transformers` |

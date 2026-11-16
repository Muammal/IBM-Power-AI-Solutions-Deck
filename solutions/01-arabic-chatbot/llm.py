"""llm.py: the model layer. Everything model-specific is read from config.yaml.
Main API: load_config(), load_model(cfg), generate(messages, stream=True)."""
import gc, os, re, threading

import yaml
from dotenv import load_dotenv
from jinja2.exceptions import TemplateError

# Arabic Unicode blocks: Arabic, Supplement, Extended-A, Presentation Forms A and B.
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_state = {}  # what load_model() loaded: cfg, backend, and the model objects


def load_config(path=None):
    load_dotenv()  # copies HF_TOKEN etc. from .env into os.environ
    path = path or os.getenv("CHATBOT_CONFIG", "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_language(text):
    """'ar' when at least 30% of the letters are Arabic script, otherwise 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for c in letters if _ARABIC.match(c))
    return "ar" if arabic / len(letters) >= 0.3 else "en"


def physical_cores():
    """Cores this process may use. PyTorch cannot detect cores on POWER and would start one
    thread per SMT hardware thread (e.g. 64 on 8 cores in SMT8), so we count them ourselves."""
    try:
        cores = set()
        for cpu in os.sched_getaffinity(0):  # one entry per hardware thread we may run on
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                cores.add(f.read().strip())     # all threads of one core share this list
        return len(cores)
    except (AttributeError, OSError):          # not Linux (e.g. a laptop): 0 = library default
        return 0


def load_model(cfg):
    unload()
    backend = cfg.get("backend", "transformers")
    threads = int((cfg.get("runtime") or {}).get("threads") or 0) or physical_cores()
    if backend == "transformers":
        # Set before torch loads: libgomp and the bundled OpenBLAS (it does the matrix
        # multiplies) read OMP_NUM_THREADS once, for every thread they create.
        if threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = cfg["model"]
        if m.get("dtype", "bfloat16") not in ("float32", "bfloat16"):
            raise ValueError("model.dtype must be float32 or bfloat16 on CPU")
        if threads:
            torch.set_num_threads(threads)  # also applies when a later load uses another value
        kw = dict(revision=m.get("revision", "main"), token=os.getenv("HF_TOKEN") or None,
                  trust_remote_code=bool(m.get("trust_remote_code", False)))
        _state["tokenizer"] = AutoTokenizer.from_pretrained(m["id"], **kw)
        _state["model"] = AutoModelForCausalLM.from_pretrained(
            m["id"], dtype=getattr(torch, m.get("dtype", "bfloat16")), **kw).eval()
        if not _state["tokenizer"].chat_template:
            raise ValueError(f"{m['id']} has no chat template. Use an -Instruct/chat model.")
    elif backend == "llamacpp":
        from llama_cpp import Llama  # only installed after Step 09
        c = cfg["llamacpp"]
        # chat_format is left unset, so llama.cpp uses the chat template stored in the GGUF file.
        _state["llm"] = Llama(model_path=c["gguf_path"], n_ctx=int(c.get("n_ctx", 4096)),
                              n_batch=int(c.get("n_batch", 512)), n_threads=threads or None,
                              verbose=False)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use transformers or llamacpp.")
    _state.update(cfg=cfg, backend=backend)


def build_messages(history):
    """Add the system prompt (in the user's language) and keep only recent turns."""
    cfg = _state["cfg"]
    if history and history[0]["role"] == "system":  # caller supplied its own system prompt
        return history
    turns = int(cfg["generation"].get("max_history_turns", 6))
    recent = history[-(2 * turns - 1):]  # last user message + (turns-1) earlier pairs
    while recent and recent[0]["role"] != "user":  # some templates (ALLaM) need user first
        recent = recent[1:]
    lang = detect_language(recent[-1]["content"])
    return [{"role": "system", "content": cfg["system_prompt"][lang].strip()}] + recent


def _without_system_role(messages):
    """Fallback for chat templates that reject a 'system' message: fold it into the first user turn."""
    if messages[0]["role"] != "system" or len(messages) < 2:
        return messages
    first = {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
    return [first] + messages[2:]


def _stream_transformers(messages, stop):
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    tok, model, g = _state["tokenizer"], _state["model"], _state["cfg"]["generation"]
    template = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True,
                    **(_state["cfg"]["model"].get("chat_template_kwargs") or {}))  # e.g. Granite enable_thinking
    try:  # the model's own chat template turns the message list into token IDs
        inputs = tok.apply_chat_template(messages, **template)
    except TemplateError:
        inputs = tok.apply_chat_template(_without_system_role(messages), **template)

    class StopOnEvent(StoppingCriteria):  # checked after every token: "should I stop now?"
        def __call__(self, input_ids, scores, **kwargs):
            return stop.is_set()

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(g["max_new_tokens"]),
                  repetition_penalty=float(g.get("repetition_penalty", 1.0)),
                  stopping_criteria=StoppingCriteriaList([StopOnEvent()]),
                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    temperature = float(g.get("temperature", 0))
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=float(g.get("top_p", 1.0)))
    else:  # greedy: also clear sampling defaults from the model's generation_config.json
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    errors = []

    def run():  # model.generate blocks, so it runs in a thread while we read the streamer
        try:
            model.generate(**kwargs)
        except Exception as exc:  # pass the error to the reader instead of hanging it
            errors.append(exc)
            streamer.end()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:  # the streamer yields "" while it waits for a complete word; skip those
        yield from (text for text in streamer if text)
    finally:  # also runs when the reader quits early (Ctrl+C): stop the model, wait for it
        stop.set()
        thread.join()
    if errors:
        raise errors[0]


def _stream_llamacpp(messages, stop):
    g = _state["cfg"]["generation"]
    opts = dict(stream=True, max_tokens=int(g["max_new_tokens"]),
                temperature=float(g.get("temperature", 0)), top_p=float(g.get("top_p", 1.0)),
                repeat_penalty=float(g.get("repetition_penalty", 1.0)))
    try:
        chunks = _state["llm"].create_chat_completion(messages=messages, **opts)
    except TemplateError:
        chunks = _state["llm"].create_chat_completion(messages=_without_system_role(messages), **opts)
    for chunk in chunks:
        if stop.is_set():
            break
        text = chunk["choices"][0]["delta"].get("content")
        if text:
            yield text


def generate(messages, stream=True, stop=None):
    """messages: [{"role": "user"|"assistant", "content": "..."}, ...] ending with a user turn.
    stop: optional threading.Event; set it from another thread to end the answer early."""
    full, stop = build_messages(messages), stop or threading.Event()
    run = _stream_transformers if _state["backend"] == "transformers" else _stream_llamacpp
    pieces = run(full, stop)
    return pieces if stream else "".join(pieces)


def count_tokens(text):
    if _state["backend"] == "transformers":
        return len(_state["tokenizer"].encode(text, add_special_tokens=False))
    return len(_state["llm"].tokenize(text.encode("utf-8"), add_bos=False))


def describe():
    cfg = _state["cfg"]
    if _state["backend"] == "llamacpp":
        return f"llamacpp | {os.path.basename(cfg['llamacpp']['gguf_path'])} | {_state['llm'].n_threads} threads"
    import torch
    return f"transformers | {cfg['model']['id']} | {cfg['model'].get('dtype')} | {torch.get_num_threads()} threads"


def unload():
    _state.clear()
    gc.collect()  # free the old weights before loading new ones

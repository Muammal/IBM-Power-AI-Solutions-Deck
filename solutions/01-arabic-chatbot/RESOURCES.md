# Solution 01 — Learning resources

## Start here

* Python for Everybody: <https://www.py4e.com/>
* Hugging Face LLM course (transformers, tokenizers, chat templates, fine-tuning): <https://huggingface.co/learn/llm-course>
* Chat templates explained (why `apply_chat_template` matters): <https://huggingface.co/docs/transformers/chat_templating>
* Text generation strategies (temperature, top-p, repetition penalty): <https://huggingface.co/docs/transformers/generation_strategies>

## Models in the presets

* Granite 4.2 (default): <https://huggingface.co/ibm-granite/granite-4.2-3b> · <https://github.com/ibm-granite/granite-4.2-language-models>
* ALLaM (Arabic-first): <https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview>
* Qwen2.5 7B: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>

## Web and streaming

* FastAPI tutorial and `StreamingResponse`: <https://fastapi.tiangolo.com/advanced/custom-response/>
* MDN: `dir="auto"` and right-to-left text in HTML: <https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/dir>

## Running LLMs on IBM Power

* ppc64le/pyeco (the install pattern and pins this solution follows): <https://github.com/ppc64le/pyeco>
* IBM Power wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel (older Power conda builds): <https://anaconda.org/rocketce>
* llama.cpp build docs (PowerPC section) and llama-cpp-python: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md> · <https://github.com/abetlen/llama-cpp-python>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>
* IBM Redbooks (search "Power10" / "Power11", processor and MMA chapters): <https://www.redbooks.ibm.com/>

## Fine-tuning

* PEFT / LoRA: <https://huggingface.co/docs/peft>
* `common/python/lora_sft.py` in this repo: a commented, CPU-only LoRA script you can read top to bottom.

## Practice exercises

1. Add a "regenerate" button to `static/index.html`.
2. Compare Granite 4.2-3B and -8B with `scripts/benchmark.py`: tokens/s and answer quality on 20 Iraqi questions.
3. Turn on `enable_thinking: true` and measure how much longer answers take on your LPAR.
4. Connect the chatbot to Solution 02's `/search` so it can answer from company documents.

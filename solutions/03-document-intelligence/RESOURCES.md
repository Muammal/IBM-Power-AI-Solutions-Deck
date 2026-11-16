# Solution 03 — Learning resources

## Fundamentals

* Python file handling and `subprocess`: <https://docs.python.org/3/library/subprocess.html>
* JSON Schema step-by-step guide: <https://json-schema.org/learn/getting-started-step-by-step>
* Pydantic and FastAPI file uploads: <https://fastapi.tiangolo.com/tutorial/request-files/>

## OCR and document AI

* Tesseract user manual (languages, page segmentation modes, improving quality): <https://tesseract-ocr.github.io/tessdoc/>
* Tesseract trained data, including `ara` and `ckb`: <https://github.com/tesseract-ocr/tessdata_best>
* Docling documentation and examples (PDF, tables, OCR options): <https://docling-project.github.io/docling/>
* Granite Docling model card: <https://huggingface.co/ibm-granite/granite-docling-258M>
* Hugging Face course on document AI (vision-language models): <https://huggingface.co/learn/computer-vision-course>

## Structured output with LLMs

* llama.cpp grammars and JSON schema constrained decoding: <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md>
* Granite 4.2 model card (prompting, tool calling): <https://huggingface.co/ibm-granite/granite-4.2-3b>

## Fine-tuning

* PEFT / LoRA documentation and conceptual guide: <https://huggingface.co/docs/peft>
* Hugging Face LLM course, fine-tuning chapter: <https://huggingface.co/learn/llm-course/chapter11/1>
* llama.cpp: converting and quantising your own models: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md> and `tools/quantize/README.md`

## IBM Power

* ppc64le/pyeco (Power wheels, examples): <https://github.com/ppc64le/pyeco>
* IBM Power wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add `schemas/customs_declaration.json` and a validator for it.
2. Add a confidence rule: if OCR text has < 60% dictionary words, force `needs_review`.
3. Build a tiny review web page that loads `out/review/*.json`, lets a user edit fields, and saves to `data/reviewed/`.
4. Compare the `basic` and `docling` engines on 20 invoices with tables; which gets line items right more often?

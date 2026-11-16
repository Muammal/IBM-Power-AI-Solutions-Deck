# Solution 02 — Learning resources

Ordered from "never done this" to "running it in production". Free unless marked.

## 1. Python and the basics

* Python for Everybody (free book + videos): <https://www.py4e.com/>
* FastAPI tutorial (official, step by step): <https://fastapi.tiangolo.com/tutorial/>
* NumPy absolute beginners guide: <https://numpy.org/doc/stable/user/absolute_beginners.html>

## 2. How RAG and embeddings work

* Hugging Face NLP Course, chapters 1–3 (transformers, tokenizers, fine-tuning): <https://huggingface.co/learn/llm-course>
* Hugging Face cookbook, "Advanced RAG": <https://huggingface.co/learn/cookbook/advanced_rag>
* Sentence embeddings and contrastive loss explained (SBERT docs, "Training Overview" and "Losses"):
  <https://sbert.net/docs/sentence_transformer/training_overview.html>
* MTEB leaderboard (compare embedding models, filter on Arabic): <https://huggingface.co/spaces/mteb/leaderboard>

## 3. The models used here

* Granite 4.2 language models (card, chat template, thinking modes): <https://huggingface.co/ibm-granite/granite-4.2-8b>
* Granite 4.2 GitHub (examples, prompts): <https://github.com/ibm-granite/granite-4.2-language-models>
* Granite Embedding models: <https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual>
* IBM Granite documentation and recipes: <https://www.ibm.com/granite/docs/> and <https://github.com/ibm-granite-community/granite-snack-cookbook>

## 4. Running models on IBM Power

* ppc64le/pyeco (IBM's Power wheel hub, install patterns, examples): <https://github.com/ppc64le/pyeco>
* IBM Power wheel index (browse packages): <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel (older Power conda builds): <https://anaconda.org/rocketce>
* llama.cpp server documentation (OpenAI API, `--jinja`, JSON schema output): <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>
* IBM Power10 MMA introduction (IBM Redbook "IBM Power10 Processor"): <https://www.redbooks.ibm.com/> (search "Power10 MMA")
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## 5. Going further

* Evaluating RAG (faithfulness, answer relevance): Ragas docs <https://docs.ragas.io/>
* Hybrid search (BM25 + vectors) and re-ranking: Pinecone learning center <https://www.pinecone.io/learn/>
* Docling for better PDF/table parsing (feeds this solution): <https://docling-project.github.io/docling/>
* Arabic NLP resources and datasets: Masader catalogue <https://arbml.github.io/masader/>

## Practice exercises

1. Add a `/ingest` endpoint that indexes one uploaded file without rebuilding everything.
2. Add BM25 keyword search and merge its results with the vector results (reciprocal rank fusion).
3. Store `department` in each chunk and filter retrieval by the caller's department.
4. Compare `granite-embedding-278m-multilingual` with `BAAI/bge-m3` on your `eval_questions.jsonl`.

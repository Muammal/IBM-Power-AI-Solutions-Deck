# Learning path: from zero to AI engineer on IBM Power

Each solution has its own `RESOURCES.md` with topic-specific material. This page is the general path.

## Stage 1 — Programming and Linux (2–6 weeks)

* Python for Everybody (free): <https://www.py4e.com/>
* The Linux Command Line (free book): <https://linuxcommand.org/tlcl.php>
* Git: Pro Git book (free, also in Arabic for early chapters): <https://git-scm.com/book>
* RHEL 9 documentation (system administration, systemd, SELinux): <https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9>

## Stage 2 — Machine learning foundations (4–8 weeks)

* Kaggle Learn micro-courses (Pandas, ML, Feature Engineering, Time Series): <https://www.kaggle.com/learn>
* Google Machine Learning Crash Course: <https://developers.google.com/machine-learning/crash-course>
* PyTorch "Learn the Basics": <https://docs.pytorch.org/tutorials/beginner/basics/intro.html>
* fast.ai Practical Deep Learning (free): <https://course.fast.ai/>

## Stage 3 — LLMs, RAG and agents (4–8 weeks)

* Hugging Face LLM Course: <https://huggingface.co/learn/llm-course>
* Hugging Face Agents Course: <https://huggingface.co/learn/agents-course>
* Hugging Face Audio Course (for Solution 04): <https://huggingface.co/learn/audio-course>
* IBM Granite documentation and community cookbooks: <https://www.ibm.com/granite/docs/> ·
  <https://github.com/ibm-granite-community>
* "Building effective agents" (Anthropic engineering): <https://www.anthropic.com/engineering/building-effective-agents>
* OWASP Top 10 for LLM applications: <https://genai.owasp.org/llm-top-10/>

## Stage 4 — Running AI on IBM Power

* **ppc64le/pyeco**: Power-optimised wheels, install patterns, tested examples: <https://github.com/ppc64le/pyeco>
* **IBM Power wheel index** (browse what is available for ppc64le): <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* **RocketCE conda channel** (Power conda builds; older versions): <https://anaconda.org/rocketce>
* llama.cpp (build options incl. PowerPC, server, quantisation): <https://github.com/ggml-org/llama.cpp>
* whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
* IBM Power Virtual Server documentation: <https://cloud.ibm.com/docs/power-iaas>
* IBM Redbooks for Power10/Power11 (processor, MMA, PowerVM, sizing): <https://www.redbooks.ibm.com/>
* IBM TechXchange community, Power and AI on Power: <https://community.ibm.com/community/user/power/home>

## Stage 5 — Arabic and Kurdish NLP

* Masader (catalogue of 600+ Arabic datasets): <https://arbml.github.io/masader/>
* CAMeL Lab tools and models (NYU Abu Dhabi): <https://github.com/CAMeL-Lab>
* Mozilla Common Voice (Arabic, Kurdish speech data): <https://commonvoice.mozilla.org/>
* KLPT, the Kurdish Language Processing Toolkit (Sorani + Kurmanji): <https://github.com/sinaahmadi/klpt>
* OPUS parallel corpora: <https://opus.nlpl.eu/>

## Stage 6 — Production and governance

* MLOps: "Made With ML" (free): <https://madewithml.com/>
* Model cards and documentation: <https://huggingface.co/docs/hub/model-cards>
* NIST AI Risk Management Framework: <https://www.nist.gov/itl/ai-risk-management-framework>

## Suggested order through this repo

05 (smallest, no LLM) → 06 (tabular ML) → 01 (first LLM) → 02 (RAG) → 07 (fine-tuning + ONNX) →
03 (documents) → 04 (speech) → 08 (translation) → 09 (agents) → 11 (OSINT services) → 10 and 12 (computer vision services).

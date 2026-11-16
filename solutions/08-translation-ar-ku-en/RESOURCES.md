# Solution 08 — Learning resources

## Machine translation basics

* Hugging Face LLM course, "Translation" (seq2seq fine-tuning): <https://huggingface.co/learn/llm-course/chapter7/4>
* Transformers translation task guide: <https://huggingface.co/docs/transformers/tasks/translation>
* chrF metric paper (Popović, 2015) and why it suits rich morphology: <https://aclanthology.org/W15-3049/>

## The model

* MADLAD-400 MT model card: <https://huggingface.co/google/madlad400-3b-mt>
* MADLAD-400 paper (dataset and models): <https://arxiv.org/abs/2309.04662>
* PEFT LoRA for seq2seq models: <https://huggingface.co/docs/peft>

## Kurdish and Arabic language resources

* Kurdish BLARK / KurdishBLARK and Sorani resources (University of Kurdistan Hewlêr, KTC corpus): <https://github.com/KurdishBLARK>
* KLPT, the Kurdish Language Processing Toolkit (Sorani + Kurmanji): <https://github.com/sinaahmadi/klpt>
* OPUS parallel corpora (filter ckb, ku, ar): <https://opus.nlpl.eu/>
* Masader Arabic dataset catalogue: <https://arbml.github.io/masader/>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a `/translate_document` endpoint that takes a `.docx`, translates each paragraph and returns a new `.docx`.
2. Compare `num_beams` 1, 2, 4 for speed and chrF.
3. Build a translation-memory lookup: exact matches from past translations are returned without the model.
4. Add Arabizi/transliteration handling for user messages written in Latin letters.

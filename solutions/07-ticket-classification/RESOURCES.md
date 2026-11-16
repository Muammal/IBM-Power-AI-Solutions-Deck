# Solution 07 — Learning resources

## Foundations

* PyTorch "Learn the Basics" (tensors, training loop): <https://docs.pytorch.org/tutorials/beginner/basics/intro.html>
* Hugging Face LLM course, chapter 3 "Fine-tuning a pretrained model": <https://huggingface.co/learn/llm-course/chapter3/1>
* Text classification task guide (transformers): <https://huggingface.co/docs/transformers/tasks/sequence_classification>

## Arabic NLP

* Masader Arabic dataset catalogue (sentiment, dialect, customer-service corpora): <https://arbml.github.io/masader/>
* CAMeL Tools (Arabic normalisation, dialect identification): <https://github.com/CAMeL-Lab/camel_tools>
* Granite Embedding model card (multilingual, Arabic): <https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual>

## Distillation and weak supervision

* "Synthetic data / LLM labelling" in the Hugging Face cookbook: <https://huggingface.co/learn/cookbook/>
* Knowledge distillation overview (Hugging Face docs, task guide): <https://huggingface.co/docs/transformers/tasks/knowledge_distillation_for_image_classification>

## ONNX and ONNX Runtime

* PyTorch → ONNX export tutorial: <https://docs.pytorch.org/tutorials/beginner/onnx/intro_onnx.html>
* ONNX Runtime quantisation guide (dynamic int8): <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
* ppc64le/pyeco ONNX example (the IBM wheels used here): <https://github.com/ppc64le/pyeco/tree/main/examples>

## IBM Power

* IBM Power wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel (onnxruntime 1.16 builds for older stacks): <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add an `urgency` head (low/medium/high) and route high urgency to a supervisor queue.
2. Measure int8 vs fp32 latency and accuracy; write the trade-off in a table.
3. Add Arabizi examples ("el net mako") and check whether the model handles them.
4. Use the confidence score to auto-close obvious "thank you" tickets.

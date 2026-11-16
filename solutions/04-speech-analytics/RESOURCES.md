# Solution 04 — Learning resources

## Audio and speech basics

* Hugging Face Audio Course (free; audio data, ASR, fine-tuning Whisper): <https://huggingface.co/learn/audio-course>
* PyAV documentation (decoding, resampling): <https://pyav.basswood-io.com/docs/stable/>
* "Fine-Tune Whisper For Multilingual ASR with Transformers" (the recipe `finetune_whisper.py` follows): <https://huggingface.co/blog/fine-tune-whisper>

## Whisper and whisper.cpp

* Whisper paper and model card: <https://huggingface.co/openai/whisper-large-v3-turbo>
* whisper.cpp (build options, server API `/inference`, model conversion, quantisation): <https://github.com/ggml-org/whisper.cpp>
* whisper.cpp server README: <https://github.com/ggml-org/whisper.cpp/tree/master/examples/server>

## Arabic and Iraqi dialect data

* Mozilla Common Voice, Arabic (CC0): <https://commonvoice.mozilla.org/ar/datasets>
* Masader, catalogue of Arabic datasets (filter: speech, Iraqi): <https://arbml.github.io/masader/>
* Arabic ASR evaluation and normalisation practices: Open Universal Arabic ASR Leaderboard <https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard>

## LLM analytics

* Granite 4.2 model card (structured output, multilingual): <https://huggingface.co/ibm-granite/granite-4.2-8b>
* llama.cpp server JSON-schema output: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheels: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add speaker separation: split stereo recordings (agent on left, customer on right) and transcribe each channel.
2. Add `agent_politeness` (1–5) to the analytics schema and validate it against 20 calls you score yourself.
3. Build a daily report: count of topics, share of angry calls, unresolved calls per agent.
4. Measure the real-time factor for `-bs 1` vs `-bs 5` and 8 vs 16 threads; plot it.

# AGENTS.md — repository-wide rules for AI agents

This file is read by AI coding agents (Claude Code, OpenAI Codex, GitHub Copilot agent, IBM watsonx Code
Assistant, Cursor, ...) before they change anything in this repository. Each solution has its own `AGENTS.md`
with solution-specific rules, its **agentic workflow** design, and its **agent change log**. A Word copy of every
AGENTS.md is in that solution's `docs/AGENTS.docx`.

## What this repository is

Twelve AI solutions for Iraqi enterprises that run on **IBM Power10/Power11 (ppc64le), CPU-only**, on IBM Cloud
Power Virtual Server or on-premises, using open, fine-tunable models. Shared platform code is in `common/`
(`common/python/vision/` for the camera solutions 10 and 12).

## Hard rules (apply to every change)

1. **Architecture: ppc64le, CPU-only.** Every runtime dependency must install as a binary wheel on
   `manylinux ppc64le / CPython 3.12` from IBM's index or PyPI. Do not add CUDA-only, x86-only, or
   compile-from-source Python packages to a `requirements.txt`. GPU code paths are optional (`--gpu`) and only for
   fine-tuning on separate, compliant machines (see `COMPLIANCE.md`).
2. **Pins.** Keep `--extra-index-url https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.03.31/+simple/`
   (with `/+simple/`) and exact pins. Moving to a newer IBM snapshot is a deliberate, repo-wide change: update
   every requirements file together and re-run `tools/check_requirements.py`.
3. **Models.** Defaults must be open-weight, fine-tunable, and Apache-2.0/MIT licensed. Check the licence on
   Hugging Face before adding any model. Never introduce non-commercial models (e.g. NLLB, `-nc` variants) as defaults.
   Model IDs belong in `config.yaml`, not in code. Update `MODELS.md` (with the verification date) when defaults change.
4. **Data stays local.** No calls to external AI APIs or telemetry. Services bind to `127.0.0.1`. Production
   services run with `HF_HUB_OFFLINE=1`.
5. **Personal data.** Never commit real documents, recordings, transactions or tickets. Never log raw personal data.
6. **Humans own consequential actions.** Agent tools that change systems, send messages, approve documents or
   move money must require human approval (see `solutions/09-it-ops-agent/src/agent.py: requires_approval`).
7. **Evaluation gates.** Every solution has an eval script. A change to models, prompts or features must not make
   its metric worse; record the before/after numbers in the change log.
8. **Surveillance limits (Solutions 10-12).** Never add face recognition, identification or tracking of individuals,
   person profiling/search, social-media account monitoring, or plate search. Keep human review before any alert
   response, notice or fine. See `COMPLIANCE.md` "Surveillance, CCTV and enforcement".
9. **Learning material.** Code is meant to be read by people learning AI on Power: keep files short, commented,
   and explain *why* on Power-specific choices (threads, MMA, SMT, wheel index).

## How to work in this repo

```bash
python tools/check_repo.py                    # py_compile all Python, bash -n all scripts, required files per solution
python tools/check_requirements.py            # resolve every requirements*.txt for ppc64le/cp312 (binary only)
python tools/build_docx.py [solutions/NN-x]   # regenerate Word files after editing Markdown
python tools/build_sizing_pdf.py             # regenerate docs/SIZING.pdf after changing models, defaults or sizing
```

Then run the **Verify** block of the solution's own `AGENTS.md`.

## Adding a new solution

1. Copy the structure of an existing one: `README.md`, `IMPLEMENTATION.md`, `RESOURCES.md`, `AGENTS.md`,
   `config.yaml`, `requirements.txt`, `src/`, `finetune/`, `eval/`, `scripts/serve.sh`.
2. `IMPLEMENTATION.md` must cover both IBM Cloud PowerVS and on-prem, each step ending with a **Check**.
3. `RESOURCES.md` must give a learning path (fundamentals → model → Power specifics → exercises) and link the
   three Power resources (pyeco, IBM wheel index, RocketCE).
4. `AGENTS.md` must have the three parts: agent brief, agentic workflow, change log.
5. Add the solution to the table in `README.md`, run the three tools above, and append to the change log.

## Agent change protocol

* Make focused changes; run the checks; regenerate docx when Markdown changes.
* Append one row to the change log of every solution you touched (date, agent/model, change, verification result).
  Do not edit or delete earlier rows.
* Repo-wide changes are logged below.

### Repository change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Created the repository: common platform layer, Solutions 02–09, Solution 01 integrated from the existing power-chatbot, compliance and model matrix, docx tooling | see "Validation status" in README.md |
| 2026-11-16 | Claude Code (Opus 5.5) | Added Solutions 10 (CCTV weapon detection), 11 (open-source situational awareness), 12 (traffic ANPR + section speed); shared `common/python/vision/` (video, detector, tracker, evidence, store, auth, train/export/eval/benchmark); surveillance rule 8; compliance section | 26 tests pass (8 + 8 + 10) on x86; RT-DETRv2 ONNX parity 1.8e-5 (dynamo exporter); live feed run; see README validation table |
| 2026-11-17 | Claude Code (Opus 5.5) | Solution 10 zero-shot mode (OWLv2) + window confirmation; shared vision fixes (file replay timing, fractional clip rate, `vision/zeroshot.py`) | public-video evaluation and live service run; see Solution 10 change log |
| 2026-11-18 | Claude Code (Opus 5.5) | Sizing guide `docs/SIZING.pdf` (9 pages) generated from the capacity model `tools/sizing_model.py` (`tools/build_sizing_pdf.py`): 12 solutions × 5 scales (small/medium/large organisation, sector-wide, country-wide), best and lean profiles, HA/DR, storage, servers, sensitivity, calibration commands | model runs; every page rendered and reviewed; numbers in the text computed from the model. Planning factors are estimates, not measured on Power |
| 2026-11-19 | Claude Code (Opus 5.5) | First run on Power10 (PowerVS s1022, 0.5 core): `common/requirements-vision-train.txt` now pins torch/torchvision `2.10.0+ppc64le2`/`0.25.0+ppc64le2` from the rolling IBM index (the snapshot's `+ppc64le1` pair fails on import: undefined `torch::autograd::_wrap_outputs`) and `onnxscript==0.7.2` (0.5.6 exports RT-DETRv2 with scores 0.19 off; the earlier 1.8e-5 x86 result had used 0.7.2); README validation row, MODELS.md note | fresh venv on Power10 from the file: imports + torchvision `nms` OK, `pip check` clean; `export_rtdetr.py` parity 0.19 (FAILED) → 5.5e-6 (OK); RT-DETRv2 290 ms/frame; Solutions 10/11/12 tests 27/27; `check_requirements.py` OK |
| 2026-11-20 | Claude Code (Opus 5.5) | `common/python/vision/video.py`: evidence-clip encoder fallback (H.264 MP4 → VP9 WebM → VP8 → MPEG-4) because IBM's ppc64le PyAV ships without libx264/mpeg4; Solution 10 clip endpoint serves the matching media type | see Solution 10 change log: `pytest` 10 passed on x86 and on Power10 |

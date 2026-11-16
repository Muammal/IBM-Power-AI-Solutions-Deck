# AGENTS.md — repository-wide rules for AI agents

This file is read by AI coding agents (Claude Code, OpenAI Codex, GitHub Copilot agent, IBM watsonx Code
Assistant, Cursor, ...) before they change anything in this repository. Each solution has its own `AGENTS.md`
with solution-specific rules, its **agentic workflow** design, and its **agent change log**. A Word copy of every
AGENTS.md is in that solution's `docs/AGENTS.docx`.

## What this repository is

Nine AI solutions for Iraqi enterprises that run on **IBM Power10/Power11 (ppc64le), CPU-only**, on IBM Cloud
Power Virtual Server or on-premises, using open, fine-tunable models. Shared platform code is in `common/`.

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
8. **Learning material.** Code is meant to be read by people learning AI on Power: keep files short, commented,
   and explain *why* on Power-specific choices (threads, MMA, SMT, wheel index).

## How to work in this repo

```bash
python tools/check_repo.py                    # py_compile all Python, bash -n all scripts, required files per solution
python tools/check_requirements.py            # resolve every requirements*.txt for ppc64le/cp312 (binary only)
python tools/build_docx.py [solutions/NN-x]   # regenerate Word files after editing Markdown
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

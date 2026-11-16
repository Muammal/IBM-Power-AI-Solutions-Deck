# Solution 08 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C3 (`medium`: 8 cores / 64 GB). C4–C5 only if you enable the
Granite post-edit.

**Check:** `common/scripts/check_power.sh` → `RESULT: PASS`.

## Step 2 — Environment and model (~12 GB download)

```bash
cd /opt/ibm-power-ai-solutions/solutions/08-translation-ar-ku-en
../../common/scripts/make_venv.sh && source venv/bin/activate
hf download google/madlad400-3b-mt
```

**Check:** `python -c "from transformers import AutoTokenizer as T; print(T.from_pretrained('google/madlad400-3b-mt').tokenize('<2ckb> سلام'))"`
shows the `<2ckb>` tag as one token.

## Step 3 — First translations

```bash
python src/translate.py --to ckb "يرجى تقديم استمارة الطلب قبل نهاية الشهر"
python src/translate.py --to ku  "Please bring your national ID card to the branch."
python src/translate.py --to ar  "Please bring your national ID card to the branch."
```

Have a native speaker of each language read the output. Note the time per sentence; `num_beams: 2` and
`dtype: bfloat16` are the main speed levers.

**Check:** output in the right script (Sorani: Arabic script; Kurmanji: Latin script) and understandable.

## Step 4 — Glossary

Fill `data/glossary.csv` with your organisation's required terms (ministry names, product names, legal terms).
`glossary_issues` in the output lists every required term that is missing.

**Check:** translate a sentence containing a glossary term; issues list is empty or names the missing term.

## Step 5 — Baseline quality

Create `data/parallel_eval.tsv` (200+ sentence pairs per direction, translated by professionals, never used for
training). Then:

```bash
python eval/eval_chrf.py data/parallel_eval.tsv
```

**Check:** chrF per direction (e.g. `ar->ckb`). This is the baseline.

## Step 6 — Fine-tune on your translation memory (LoRA, CPU)

Export aligned sentence pairs from past professional translations into `data/parallel_train.tsv` (same columns).

```bash
python finetune/lora_madlad.py --train data/parallel_train.tsv --eval data/parallel_eval.tsv --epochs 2
sed -i 's#^  adapter: null#  adapter: out/lora-madlad/adapter#' config.yaml
python eval/eval_chrf.py data/parallel_eval.tsv
```

Start with 2,000 pairs to measure time per epoch on your LPAR, then scale. GPU option: `--gpu` on a compliant
machine ([`COMPLIANCE.md`](../../COMPLIANCE.md)); copy `out/lora-madlad/adapter` back.

**Check:** chrF ≥ baseline in every direction you trained; native-speaker review agrees.

## Step 7 — Optional Granite post-edit

Start Granite (C5) and set `postedit.enabled: true`. It only runs for Arabic/English targets **and** only when a
glossary term is missing.

**Check:** a sentence with a missing glossary term comes back with `"postedited": true` and the term fixed.

## Step 8 — Production

```bash
sudo systemctl enable --now ai-app@08-translation-ar-ku-en
curl -s localhost:8008/translate -H 'Content-Type: application/json' \
     -d '{"text":"يرجى مراجعة الفرع لتحديث بياناتك","target":"ckb"}' | jq
```

* Official documents: machine translation + professional review. Never publish legal text unreviewed.
* For documents: convert with Solution 03 (`src/convert.py`), translate paragraph by paragraph, rebuild.

**Check:** `/health` lists the four languages and the adapter in use.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Output in the wrong language | missing/wrong `<2xx>` tag | target must be `ar`, `ckb`, `ku` or `en` |
| Truncated long sentences | `max_new_tokens` | raise it; very long sentences: split on `،` / `;` |
| Slow (> 10 s per sentence) | beams / fp32 | `num_beams: 2`, `dtype: bfloat16`, SMT4, physical-core threads |
| Kurmanji quality poor | little training data in MADLAD | Step 6 with your Kurmanji translations matters most here |

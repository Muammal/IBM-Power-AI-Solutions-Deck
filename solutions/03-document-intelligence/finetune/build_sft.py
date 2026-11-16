"""build_sft.py: turn human-corrected extractions into LoRA training data for Granite 4.2-3B.

Input : data/reviewed/*.json, each the output of src/extract.py where a reviewer fixed "data" by hand
        (the "text" field must still be present: it is the model input).
Output: data/sft_train.jsonl, data/sft_eval.jsonl in the chat format used by common/python/lora_sft.py

    python finetune/build_sft.py
    python ../../common/python/lora_sft.py --data data/sft_train.jsonl --eval-data data/sft_eval.jsonl \
        --model ibm-granite/granite-4.2-3b --out out/lora-invoice --epochs 3 --merge
    ../../common/scripts/hf_to_gguf.sh out/lora-invoice/merged /opt/models/granite-4.2-3b-invoice Q8_0
Aim for 200+ reviewed documents, covering every vendor layout you see often.
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extract import SYSTEM  # noqa: E402


def main():
    files = sorted((ROOT / "data" / "reviewed").glob("*.json"))
    random.seed(7)
    random.shuffle(files)
    n_eval = max(1, len(files) // 10)
    for name, part in (("sft_eval.jsonl", files[:n_eval]), ("sft_train.jsonl", files[n_eval:])):
        with open(ROOT / "data" / name, "w", encoding="utf-8") as out:
            for f in part:
                r = json.loads(f.read_text(encoding="utf-8"))
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Document type: {r['document_type']}\n\nDocument text:\n{r['text'][:24000]}"},
                        {"role": "assistant", "content": json.dumps(r["data"], ensure_ascii=False)}]
                out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} examples")


if __name__ == "__main__":
    main()

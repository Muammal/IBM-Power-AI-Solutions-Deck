"""build_sft.py: analyst corrections -> LoRA training data for the event extractor (Granite 4.2-3B or -8B).

Every correction made in the dashboard/API (POST /api/events/{id}/correction) is a labelled example. With 300+:
    python finetune/build_sft.py                        # -> data/sft_events.jsonl
    python ../../common/python/lora_sft.py --data data/sft_events.jsonl --model ibm-granite/granite-4.2-3b \
        --out out/lora-events --epochs 2 --merge
    ../../common/scripts/hf_to_gguf.sh out/lora-events/merged /opt/models/granite-4.2-3b-events Q8_0
Then run eval/eval_extraction.py against the new llama-server before switching LLM_URL.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG  # noqa: E402
from engine import OsintStore  # noqa: E402
from enrich import SYSTEM  # noqa: E402


def main():
    store = OsintStore(ROOT / CFG["storage"]["db"])
    rows = store.q("SELECT e.*, i.text FROM events e JOIN items i ON i.id = e.item_id WHERE e.corrected IS NOT NULL")
    out = ROOT / "data" / "sft_events.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            fix = json.loads(r["corrected"])
            target = {"is_event": fix.get("is_event", True), "category": fix.get("category", r["category"]),
                      "governorates": fix.get("governorates", json.loads(r["governorates"])),
                      "location_text": r["location_text"], "severity": fix.get("severity", r["severity"]),
                      "organizations": json.loads(r["organizations"]), "summary_ar": r["summary_ar"],
                      "summary_en": r["summary_en"], "confidence": 1.0}
            user = f"Title: {r['title']}\nText: {(r['text'] or '')[:3000]}\nSource: {r['source']}"
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                                             {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]},
                               ensure_ascii=False) + "\n")
    print(f"{len(rows)} corrected events -> {out}")


if __name__ == "__main__":
    main()

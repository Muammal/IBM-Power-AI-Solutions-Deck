"""translate.py: MADLAD-400 translation + glossary check + optional Granite post-edit.

    python src/translate.py --to ckb "يرجى تقديم الطلب قبل نهاية الشهر"
    python src/translate.py --to en  "Daxwazname heta dawiya mehê bişînin"
"""
import argparse
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["runtime"]["threads"])

import torch  # noqa: E402
from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: E402

from llm_client import LLM  # noqa: E402

_SENT = re.compile(r"(?<=[.!?؟。\n])\s+")


class Translator:
    def __init__(self, cfg=CFG):
        m = cfg["model"]
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(m["id"])
        self.model = T5ForConditionalGeneration.from_pretrained(m["id"], dtype=getattr(torch, m["dtype"])).eval()
        if m.get("adapter"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(ROOT / m["adapter"])).merge_and_unload().eval()
        self.glossary = self._load_glossary(ROOT / cfg["glossary_csv"])

    @staticmethod
    def _load_glossary(path):
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @torch.inference_mode()
    def translate(self, text, target):
        if target not in self.cfg["languages"]:
            raise ValueError(f"target must be one of {list(self.cfg['languages'])}")
        m = self.cfg["model"]
        # Translate sentence by sentence: T5 models are trained on sentence pairs; long inputs degrade.
        sents = [s for s in _SENT.split(text.strip()) if s]
        enc = self.tok([f"<2{target}> {s}" for s in sents], return_tensors="pt", padding=True)
        out = self.model.generate(**enc, max_new_tokens=int(m["max_new_tokens"]), num_beams=int(m["num_beams"]))
        result = " ".join(self.tok.batch_decode(out, skip_special_tokens=True))
        return {"translation": result, "glossary_issues": self.check_glossary(text, result, target)}

    def check_glossary(self, source, translation, target):
        """Required terms: if a glossary source term appears in the source, its target term must appear."""
        issues = []
        for row in self.glossary:
            tgt = row.get(f"term_{target}")
            for lang, term in row.items():
                if term and lang != f"term_{target}" and term.lower() in source.lower() and tgt and tgt not in translation:
                    issues.append({"source_term": term, "expected": tgt})
                    break
        return issues

    def postedit(self, source, result, target):
        if not (self.cfg["postedit"]["enabled"] and target in ("ar", "en") and result["glossary_issues"]):
            return result
        terms = "; ".join(f"{i['source_term']} -> {i['expected']}" for i in result["glossary_issues"])
        fixed = LLM(self.cfg["postedit"]["base_url"]).chat([{"role": "user", "content":
            f"Correct this {self.cfg['languages'][target]} translation so it uses the required terms ({terms}). "
            f"Change nothing else. Return only the corrected translation.\nSource: {source}\nTranslation: {result['translation']}"}],
            temperature=0.0, max_tokens=800)
        return {"translation": fixed, "glossary_issues": self.check_glossary(source, fixed, target), "postedited": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    t = Translator()
    src = " ".join(a.text)
    print(t.postedit(src, t.translate(src, a.to), a.to))

"""extract.py: document text -> JSON that matches schemas/<type>.json -> business-rule validation.

The LLM output is constrained by the JSON schema (llama-server grammar), so it always parses. Validation then
decides: "auto_approved" or "needs_review" (with reasons), the hand-off point to a human or an agent.

    python src/extract.py data/samples/invoice_001.md
    python src/extract.py scans/INV-2291.pdf --type invoice
"""
import argparse
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from convert import convert  # noqa: E402
from llm_client import LLM  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
SYSTEM = """You extract data from business documents (Arabic or English, often OCR output with errors).
Rules:
- Copy names exactly as written (keep Arabic names in Arabic script).
- Dates as YYYY-MM-DD. Numbers as plain numbers without thousands separators.
- Currency: "دينار"/"د.ع"/"IQD" -> IQD; "$"/"دولار"/"USD" -> USD.
- If a field is not present, use null. Never guess values that are not in the text."""


def load_schema(doc_type):
    return json.loads((ROOT / "schemas" / f"{doc_type}.json").read_text(encoding="utf-8"))


def validate_invoice(d, tol, currencies):
    """Business rules. Returns a list of human-readable problems (empty = OK)."""
    problems = []
    try:
        dt.date.fromisoformat(d.get("invoice_date") or "")
    except ValueError:
        problems.append(f"invoice_date '{d.get('invoice_date')}' is not a valid YYYY-MM-DD date")
    if d.get("currency") not in currencies:
        problems.append(f"currency '{d.get('currency')}' not in {currencies}")
    items = d.get("line_items") or []
    for i, it in enumerate(items, 1):
        if abs(it["quantity"] * it["unit_price"] - it["amount"]) > tol * max(1, abs(it["amount"])):
            problems.append(f"line {i}: quantity x unit_price != amount")
    items_sum = sum(it["amount"] for it in items)
    base = d.get("subtotal") if d.get("subtotal") is not None else items_sum
    if items and abs(items_sum - base) > tol * max(1, base):
        problems.append(f"line items sum {items_sum:,.2f} != subtotal {base:,.2f}")
    if abs(base + (d.get("tax") or 0) - d["total"]) > tol * max(1, d["total"]):
        problems.append(f"subtotal + tax != total {d['total']:,.2f}")
    return problems


def extract(path, doc_type=None, llm=None):
    doc_type = doc_type or CFG["document_type"]
    c = CFG["converter"]
    text = convert(path, c["engine"], c["ocr_languages"], c["ocr_dpi"])
    llm = llm or LLM(CFG["llm"]["base_url"])
    data = llm.chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Document type: {doc_type}\n\nDocument text:\n{text[:24000]}"}],
                    json_schema=load_schema(doc_type), temperature=CFG["llm"]["temperature"],
                    max_tokens=CFG["llm"]["max_tokens"])
    v = CFG["validation"]
    problems = validate_invoice(data, v["amount_tolerance"], v["currencies"]) if doc_type == "invoice" else []
    return {"file": str(path), "document_type": doc_type, "data": data,
            "status": "needs_review" if problems else "auto_approved", "problems": problems, "text": text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--type")
    a = ap.parse_args()
    out = extract(a.file, a.type)
    out.pop("text")
    print(json.dumps(out, ensure_ascii=False, indent=2))

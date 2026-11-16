"""eval_chrf.py: chrF++-style character n-gram F-score (no extra dependencies) on a held-out TSV.

chrF works better than BLEU for morphologically rich languages like Arabic and Kurdish.
    python eval/eval_chrf.py data/parallel_eval.tsv              # base model / current adapter in config.yaml
"""
import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _ngrams(text, n):
    s = text.replace(" ", "")
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(refs, hyps, n=6, beta=2.0):
    prec, rec = [], []
    for k in range(1, n + 1):
        match = hyp_total = ref_total = 0
        for r, h in zip(refs, hyps):
            rn, hn = _ngrams(r, k), _ngrams(h, k)
            match += sum((rn & hn).values())
            hyp_total += sum(hn.values())
            ref_total += sum(rn.values())
        prec.append(match / hyp_total if hyp_total else 0)
        rec.append(match / ref_total if ref_total else 0)
    p, r = sum(prec) / n, sum(rec) / n
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def main():
    from translate import Translator
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "parallel_sample.tsv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    t = Translator()
    by_pair = collections.defaultdict(lambda: ([], []))
    for r in rows:
        hyp = t.translate(r["src"], r["tgt_lang"])["translation"]
        refs, hyps = by_pair[(r["src_lang"], r["tgt_lang"])]
        refs.append(r["tgt"])
        hyps.append(hyp)
    for (s, tg), (refs, hyps) in sorted(by_pair.items()):
        print(f"{s}->{tg}: n={len(refs)} chrF={chrf(refs, hyps):.1f}")


if __name__ == "__main__":
    main()

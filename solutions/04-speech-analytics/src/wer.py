"""wer.py: word error rate with Arabic normalisation, no extra dependencies.

Arabic WER is inflated by spelling variants that mean the same thing (أ/إ/آ/ا, ة/ه, ى/ي, diacritics, tatweel).
We normalise those before comparing, which is standard practice for Arabic ASR evaluation.
"""
import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat + tatweel
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text):
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = _PUNCT.sub(" ", text.lower())
    return text.split()


def _edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs, hyps):
    """Corpus WER: total word edits / total reference words."""
    errors = words = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize(r), normalize(h)
        errors += _edits(rw, hw)
        words += len(rw)
    return errors / max(1, words)


if __name__ == "__main__":
    print(wer(["السلام عليكم شلونك"], ["السلام عليكم شلونك"]))   # 0.0
    print(wer(["أريد أن أدفع الفاتورة"], ["اريد ان ادفع فاتوره"]))    # 0.25 (one word missing "ال")

"""plates.py: plate text normalisation, validation and voting.

Iraqi plates mix Arabic-Indic digits (٠-٩), Latin letters and Arabic text (governorate names). The OCR model is
trained to output the normalised alphabet below; everything is reduced to ASCII digits + Latin capitals so that
the same vehicle read by two cameras produces the same string.
"""
import collections
import hashlib
import hmac
import re

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"           # OCR output classes (+ CTC blank)
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Letters OCR commonly confuses with digits on plates: only applied where the format needs a digit.
_TO_DIGIT = str.maketrans("OQDIZSB", "0001258")


def normalise(text):
    t = (text or "").translate(_DIGITS).upper()
    return re.sub(r"[^0-9A-Z]", "", t)


def valid(text, formats):
    return any(re.fullmatch(f, text) for f in formats)


def repair(text, formats):
    """Try the digit-confusion fix when the raw text matches no format (e.g. 'I23456' -> '123456')."""
    if valid(text, formats):
        return text
    fixed = text.translate(_TO_DIGIT)
    return fixed if valid(fixed, formats) else None


def is_ambiguous(text, formats):
    """True when the text AND its digit-confusion variant are both valid plates (e.g. 'I23456' vs '123456' when
    one format allows a leading letter). Such a read may be a different vehicle: never use it for enforcement."""
    if not text:
        return False
    fixed = text.translate(_TO_DIGIT)
    return fixed != text and valid(text, formats) and valid(fixed, formats)


def vote(reads):
    """reads: [(text, confidence)] from several crops of the same vehicle -> (text, confidence, agreement)."""
    reads = [(t, c) for t, c in reads if t]
    if not reads:
        return None, 0.0, 0.0
    score = collections.defaultdict(float)
    for t, c in reads:
        score[t] += c
    best = max(score, key=score.get)
    agreement = sum(1 for t, _ in reads if t == best) / len(reads)
    conf = max(c for t, c in reads if t == best)
    return best, conf, agreement


def plate_key(text, key):
    """Keyed hash: lets statistics follow a vehicle between cameras without storing the plate."""
    return hmac.new(key.encode(), text.encode(), hashlib.sha256).hexdigest()[:32]

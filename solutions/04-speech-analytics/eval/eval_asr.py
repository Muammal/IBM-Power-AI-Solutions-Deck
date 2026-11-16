"""eval_asr.py: WER of the running whisper-server on a labelled test folder.

    python eval/eval_asr.py data/asr/test                  # metadata.csv: file_name,transcription
Run it with the stock model and with your fine-tuned model (restart whisper-server with the other .bin).
"""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import transcribe  # noqa: E402
from wer import wer  # noqa: E402


def main():
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "asr" / "test")
    with open(folder / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    refs, hyps, total_s = [], [], 0.0
    for r in rows:
        text, sec = transcribe(folder / r["file_name"])
        refs.append(r["transcription"])
        hyps.append(text)
        total_s += sec
        print(f"{r['file_name']}: {wer([r['transcription']], [text]):.2f}")
    print(f"\nclips={len(rows)} audio={total_s / 60:.1f} min  WER={wer(refs, hyps):.3f}")


if __name__ == "__main__":
    main()

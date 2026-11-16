"""finetune_whisper.py: adapt Whisper to Iraqi Arabic (your accents, product names, call-center vocabulary).

Data folder layout (consent-recorded calls or read speech; 10-50 hours is a good start):
    data/asr/train/metadata.csv   columns: file_name,transcription
    data/asr/train/*.wav|mp3|m4a
    data/asr/test/metadata.csv  (+ audio)

    python finetune/finetune_whisper.py --model openai/whisper-small --epochs 3
    # -> out/whisper-iraqi ; then convert for whisper.cpp (IMPLEMENTATION.md Step 8)

CPU guide (8 Power10 cores, measure yours): whisper-small (244M) ~ 1 h of audio per 1-2 h of training per epoch.
whisper-large-v3-turbo (809M) is ~4x slower: use it on CPU only for small datasets, or --gpu on a compliant GPU.
"""
import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads  # noqa: E402

configure_threads()

import torch  # noqa: E402
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,  # noqa: E402
                          WhisperProcessor)

from audio import to_float32, to_pcm16k  # noqa: E402
from wer import wer  # noqa: E402


class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, folder, processor, max_seconds=30):
        folder = pathlib.Path(folder)
        with open(folder / "metadata.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.items = []
        for r in rows:  # decode once up front (simple; for >50 h of audio cache features to disk instead)
            pcm, sec = to_pcm16k(str(folder / r["file_name"]))
            if 0 < sec <= max_seconds:
                feats = processor.feature_extractor(to_float32(pcm), sampling_rate=16000).input_features[0]
                labels = processor.tokenizer(r["transcription"]).input_ids
                self.items.append({"input_features": feats, "labels": labels, "text": r["transcription"]})
        print(f"{folder}: {len(self.items)} clips (<= {max_seconds}s)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(processor, decoder_start_id):
    def collate(batch):
        feats = processor.feature_extractor.pad([{"input_features": b["input_features"]} for b in batch],
                                                return_tensors="pt")
        labels = processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == decoder_start_id).all():  # the model adds the start token itself
            ids = ids[:, 1:]
        return {"input_features": feats["input_features"], "labels": ids}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--train", default=str(ROOT / "data" / "asr" / "train"))
    ap.add_argument("--test", default=str(ROOT / "data" / "asr" / "test"))
    ap.add_argument("--out", default=str(ROOT / "out" / "whisper-iraqi"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu", action="store_true", help="one CUDA device (see COMPLIANCE.md)")
    a = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(a.model, language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.model)
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train, test = AudioFolder(a.train, processor), AudioFolder(a.test, processor)

    def metrics(pred):
        ids = pred.label_ids.copy()
        ids[ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        ref = processor.batch_decode(ids, skip_special_tokens=True)
        return {"wer": wer(ref, hyp)}

    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch, gradient_accumulation_steps=2,
        learning_rate=a.lr, warmup_ratio=0.1, num_train_epochs=a.epochs, gradient_checkpointing=True,
        bf16=True, use_cpu=not a.gpu, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        predict_with_generate=True, generation_max_length=225, logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False, report_to=[])
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=test,
                             data_collator=make_collator(processor, model.config.decoder_start_token_id),
                             compute_metrics=metrics, processing_class=processor)
    print("baseline:", trainer.evaluate())
    trainer.train()
    print("fine-tuned:", trainer.evaluate())
    trainer.save_model(a.out)
    processor.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()

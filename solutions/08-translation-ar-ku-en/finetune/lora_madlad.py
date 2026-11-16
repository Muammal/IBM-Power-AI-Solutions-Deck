"""lora_madlad.py: LoRA fine-tuning of MADLAD-400-3B on your parallel sentences (CPU, bf16).

Data: TSV with header  src_lang  tgt_lang  src  tgt   (e.g. ar  ckb  <Arabic sentence>  <Sorani sentence>)
Include both directions you need. 5,000-50,000 aligned sentence pairs from your own translated documents
(forms, notices, contracts) is a good range.

    python finetune/lora_madlad.py --train data/parallel_train.tsv --eval data/parallel_eval.tsv --epochs 2
    # then config.yaml: model.adapter: out/lora-madlad/adapter
Time: LoRA r=16 on 3B, 10k pairs x 1 epoch ~ several hours on 8 Power10 cores. Start with 2k pairs to measure.
"""
import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["runtime"]["threads"])

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,  # noqa: E402
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration)


class Pairs(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=256):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.items = [tok(f"<2{r['tgt_lang']}> {r['src']}", text_target=r["tgt"], max_length=max_len, truncation=True)
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--out", default=str(ROOT / "out" / "lora-madlad"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu", action="store_true", help="one CUDA device (see COMPLIANCE.md)")
    a = ap.parse_args()

    mid = CFG["model"]["id"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = T5ForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             task_type="SEQ_2_SEQ_LM",
                                             target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"]))
    model.print_trainable_parameters()

    train = Pairs(a.train, tok)
    evals = Pairs(a.eval, tok) if a.eval else None
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=4, warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=10,
        eval_strategy="epoch" if evals else "no", save_strategy="epoch", save_total_limit=2,
        bf16=True, use_cpu=not a.gpu, report_to=[])
    Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=evals,
                   data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    model.save_pretrained(pathlib.Path(a.out) / "adapter")
    print("adapter saved ->", pathlib.Path(a.out) / "adapter", "| set model.adapter in config.yaml, run eval")


if __name__ == "__main__":
    main()

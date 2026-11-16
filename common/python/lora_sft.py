"""lora_sft.py: LoRA supervised fine-tuning of a chat model (Granite 4.2 by default) on CPU.

Used by Solutions 02, 03 and 09 (and works for Solution 01's chatbot). Runs on Power10/Power11 with
IBM's torch wheel: bf16 matrix multiplies go through OpenBLAS POWER10 kernels that use MMA.

Data: JSONL, one conversation per line, the LAST message is the assistant answer to learn:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

    python common/python/lora_sft.py --data train.jsonl --out out/lora \
        --model ibm-granite/granite-4.2-3b --epochs 2 --merge
Then:  common/scripts/hf_to_gguf.sh out/lora/merged /opt/models/my-model Q4_0

Sizing on CPU (rule of thumb, measure on your LPAR): 3B + LoRA r=16, 1,000 examples x 512 tokens,
1 epoch ~ 1-3 hours on 8 Power10 cores. 8B is ~3x slower. Start with 3B.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from power_runtime import configure_threads  # noqa: E402

configure_threads()  # before torch import

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,  # noqa: E402
                          TrainingArguments)

TEMPLATE_KW = {"enable_thinking": False}  # Granite 4.2: train the direct-answer mode


def encode(tok, messages, max_len):
    """Token ids for the whole conversation; labels = -100 everywhere except the final answer."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
    full = tok.apply_chat_template(messages, tokenize=False, **TEMPLATE_KW)
    if not full.startswith(prompt):  # template renders history differently: fall back to concatenation
        full = prompt + messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[:len(f_ids)]}


class JsonlChat(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.items = [encode(tok, r["messages"], max_len) for r in rows]
        self.items = [x for x in self.items if any(l != -100 for l in x["labels"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--merge", action="store_true", help="also save base+adapter merged (for GGUF)")
    ap.add_argument("--gpu", action="store_true", help="train on one CUDA GPU (see COMPLIANCE.md)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = JsonlChat(a.data, tok, a.max_len)
    evals = JsonlChat(a.eval_data, tok, a.max_len) if a.eval_data else None
    print(f"train examples: {len(train)}" + (f" | eval: {len(evals)}" if evals else ""))

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=5,
        eval_strategy="epoch" if evals else "no", save_strategy="epoch", save_total_limit=2,
        bf16=True, use_cpu=not a.gpu, dataloader_num_workers=0, report_to=[])
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=evals,
            data_collator=collate(pad_id)).train()

    model.save_pretrained(os.path.join(a.out, "adapter"))
    tok.save_pretrained(os.path.join(a.out, "adapter"))
    print("adapter saved:", os.path.join(a.out, "adapter"))
    if a.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(a.out, "merged"), safe_serialization=True)
        tok.save_pretrained(os.path.join(a.out, "merged"))
        print("merged model saved:", os.path.join(a.out, "merged"))


if __name__ == "__main__":
    main()

"""finetune_ttm.py: fine-tune Granite TTM-R2 on your series (CPU, minutes).

With finetune.use_exogenous: true the model also learns how known drivers (here temperature) move the target,
through TTM's channel-mixing decoder ("exogenous mixing"), following IBM's TTM fine-tuning recipe.
The pre-trained backbone is frozen; only the small decoder/head is trained.

    python finetune/finetune_ttm.py
    python src/forecast.py --finetuned          # compare test_mae with zero-shot
"""
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads  # noqa: E402

configure_threads()

import torch  # noqa: E402
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments  # noqa: E402
from tsfm_public import get_datasets, get_model  # noqa: E402

from forecast import CFG, load_df, make_preprocessor  # noqa: E402


def main():
    m, d, f = CFG["model"], CFG["data"], CFG["finetune"]
    use_exog = bool(f["use_exogenous"] and d["control_columns"])
    tsp = make_preprocessor(use_exog)
    train, valid, test = get_datasets(tsp, load_df(), split_config=d["split"])

    kw = {}
    if use_exog:
        kw = dict(num_input_channels=tsp.num_input_channels, decoder_mode="mix_channel",
                  prediction_channel_indices=tsp.prediction_channel_indices,
                  exogenous_channel_indices=tsp.exogenous_channel_indices,
                  fcm_context_length=1, fcm_use_mixer=True, fcm_mix_layers=2,
                  enable_forecast_channel_mixing=True, fcm_prepend_past=True)
    model = get_model(m["id"], context_length=m["context_length"], prediction_length=m["prediction_length"], **kw)
    for p in model.backbone.parameters():  # keep the pre-trained knowledge, train the decoder/head only
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"train windows={len(train)} valid={len(valid)} trainable params={trainable:,} exogenous={use_exog}")

    out = ROOT / m["finetuned_path"]
    bs, epochs = int(f["batch_size"]), int(f["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(f["learning_rate"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, float(f["learning_rate"]), epochs=epochs,
                                                steps_per_epoch=math.ceil(len(train) / bs))
    args = TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=epochs,
                             per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                             load_best_model_at_end=True, metric_for_best_model="eval_loss",
                             greater_is_better=False, use_cpu=True, dataloader_num_workers=0, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid, optimizers=(opt, sched),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    trainer.train()
    print("test:", trainer.evaluate(test))
    trainer.save_model(str(out))
    tsp.save_pretrained(str(out))  # writes preprocessor_config.json (scalers, columns) next to the model
    print("saved", out)


if __name__ == "__main__":
    main()

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import yaml
from rouge_score import rouge_scorer as rouge_scorer_lib
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)

class ProgressCallback(TrainerCallback):

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_start = None
        self.run_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        current_epoch = int(state.epoch) + 1 if state.epoch is not None else "?"
        print(f"\n{'='*60}")
        print(f"  EPOCH {current_epoch} / {self.total_epochs} STARTING")
        print(f"{'='*60}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.epoch is None:
            return

        if "loss" in logs and "eval_loss" not in logs:
            elapsed = time.time() - self.run_start
            steps_done = state.global_step
            total_steps = state.max_steps
            if total_steps and steps_done:
                eta = (elapsed / steps_done) * (total_steps - steps_done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            else:
                eta_str = "?"
            print(
                f"  Step {steps_done:>4}/{total_steps} | "
                f"Loss: {logs.get('loss', 0):.4f} | "
                f"LR: {logs.get('learning_rate', 0):.2e} | "
                f"ETA: {eta_str}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = round(state.epoch or 0)
        epoch_time = time.time() - (self.epoch_start or self.run_start)
        total_elapsed = time.time() - self.run_start

        print(f"\n  --- Epoch {epoch} Evaluation Results ---")
        print(f"  Eval Loss  : {metrics.get('eval_loss', 0):.4f}")
        print(f"  ROUGE-1    : {metrics.get('eval_rouge1', 0):.4f}")
        print(f"  ROUGE-2    : {metrics.get('eval_rouge2', 0):.4f}")
        print(f"  ROUGE-L    : {metrics.get('eval_rougeL', 0):.4f}")
        print(f"  Epoch time : {epoch_time:.1f}s")
        print(f"  Total time : {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")

        epochs_done = epoch
        epochs_left = self.total_epochs - epochs_done
        if epochs_done > 0:
            avg_epoch_time = total_elapsed / epochs_done
            eta = avg_epoch_time * epochs_left
            print(f"  ETA        : ~{time.strftime('%H:%M:%S', time.gmtime(eta))} remaining")
        print(f"{'='*60}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    # 1. Load config from YAML
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    gen_cfg = cfg["generation"]

    model_name = model_cfg["name_or_path"]
    model_type = model_cfg["type"]  
    source_prefix = data_cfg.get("source_prefix", "") 

    print(f"Model     : {model_name}")
    print(f"Type      : {model_type}")
    print(f"Output dir: {train_cfg['output_dir']}")
    print(f"Device    : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (no GPU detected)'}")

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. Load Silver Dataset CSVs directly via pandas (avoids fsspec cache issues)
    train_df = pd.read_csv(data_cfg["train_file"])
    val_df = pd.read_csv(data_cfg["validation_file"])
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False)
    })

    # 4. Tokenization
    max_src = data_cfg["max_source_length"]
    max_tgt = data_cfg["max_target_length"]

    def preprocess_function(examples):
        inputs = examples["factual"]

        # T5 requires a task prefix; BART does not
        if source_prefix:
            inputs = [source_prefix + inp for inp in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=max_src,
            truncation=True,
            padding=False
        )

        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["satirical"],
                max_length=max_tgt,
                truncation=True,
                padding=False
            )

        # Replace pad token id with -100 so loss ignores padding
        label_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 5. Data Collator — handles dynamic padding per batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # 6. ROUGE metric for per-epoch evaluation
    _rouge_scorer = rouge_scorer_lib.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels (padding) before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        r1, r2, rl = [], [], []
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = _rouge_scorer.score(label, pred)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': round(float(np.mean(r1)), 4),
            'rouge2': round(float(np.mean(r2)), 4),
            'rougeL': round(float(np.mean(rl)), 4),
        }

    # 7. Training Arguments — all values driven by config
    use_fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        label_smoothing_factor=train_cfg.get("label_smoothing_factor", 0.0),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "epoch")),
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        predict_with_generate=train_cfg["predict_with_generate"],
        generation_num_beams=gen_cfg["num_beams"],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=use_fp16,
        report_to=train_cfg.get("report_to", "none")
    )

    # 8. Initialize Trainer
    num_epochs = train_cfg["num_train_epochs"]
    batch_size = train_cfg["per_device_train_batch_size"]
    total_steps = (len(tokenized_datasets["train"]) // batch_size) * num_epochs
    gpu_eta = total_steps * 0.7  # ~0.7s per step on GPU
    cpu_eta = total_steps * 8.0  # ~8s per step on CPU
    print(f"\nTraining estimate: {total_steps} steps total ({num_epochs} epochs x {total_steps // num_epochs} steps/epoch)")
    if torch.cuda.is_available():
        print(f"Estimated time  : ~{time.strftime('%H:%M:%S', time.gmtime(gpu_eta))} on GPU")
    else:
        print(f"Estimated time  : ~{time.strftime('%H:%M:%S', time.gmtime(cpu_eta))} on CPU (consider using a GPU)")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressCallback(total_epochs=num_epochs)
        ]
    )

    # 9. Train
    print("\nStarting training...")
    trainer.train()

    # 10. Save best model
    final_output = os.path.join(train_cfg["output_dir"], "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\nBest model saved to: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART or T5 on the silver satire dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bart_training.yaml",
        help="Path to training config YAML (bart_training.yaml or t5_training.yaml)."
    )
    args = parser.parse_args()
    main(args)
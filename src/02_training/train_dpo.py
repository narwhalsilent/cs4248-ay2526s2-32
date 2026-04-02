import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PreferenceDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


@dataclass
class PreferenceCollator:
    tokenizer: AutoTokenizer
    max_source_length: int
    max_target_length: int
    use_prefixed_prompt: bool = True

    def _tokenize_targets(self, texts):
        encoded = self.tokenizer(
            text_target=texts,
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"]
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return labels

    def __call__(self, batch):
        prompts = []
        chosen = []
        rejected = []

        for record in batch:
            if self.use_prefixed_prompt and record.get("prompt_with_prefix"):
                prompts.append(record["prompt_with_prefix"])
            else:
                prompts.append(record["prompt"])
            chosen.append(record["chosen"])
            rejected.append(record["rejected"])

        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": prompt_tokens["input_ids"],
            "attention_mask": prompt_tokens["attention_mask"],
            "chosen_labels": self._tokenize_targets(chosen),
            "rejected_labels": self._tokenize_targets(rejected),
        }


def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    logits = outputs.logits
    valid_mask = labels != -100
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logps = token_logps * valid_mask
    return token_logps.sum(dim=-1)


def dpo_loss(policy_model, reference_model, batch, beta, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    policy_chosen = sequence_logprob(policy_model, input_ids, attention_mask, chosen_labels)
    policy_rejected = sequence_logprob(policy_model, input_ids, attention_mask, rejected_labels)
    with torch.no_grad():
        ref_chosen = sequence_logprob(reference_model, input_ids, attention_mask, chosen_labels)
        ref_rejected = sequence_logprob(reference_model, input_ids, attention_mask, rejected_labels)

    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    losses = -F.logsigmoid(logits)
    accuracy = (logits > 0).float().mean()
    return losses.mean(), accuracy.item()


def evaluate(model, reference_model, dataloader, beta, device):
    model.eval()
    loss_total = 0.0
    accuracy_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, accuracy = dpo_loss(model, reference_model, batch, beta, device)
            loss_total += loss.item()
            accuracy_total += accuracy
            num_batches += 1

    if num_batches == 0:
        return {"loss": math.nan, "accuracy": math.nan}

    return {
        "loss": loss_total / num_batches,
        "accuracy": accuracy_total / num_batches,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = load_jsonl(args.train_preferences)
    val_records = load_jsonl(args.validation_preferences) if args.validation_preferences else []
    if not train_records:
        raise ValueError(f"No training preferences found in {args.train_preferences}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    policy_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    reference_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.reference_model_path or args.model_path
    ).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    collator = PreferenceCollator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        use_prefixed_prompt=not args.use_raw_prompt_only,
    )

    train_loader = DataLoader(
        PreferenceDataset(train_records),
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = None
    if val_records:
        val_loader = DataLoader(
            PreferenceDataset(val_records),
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    optimizer = AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = float("-inf")
    best_dir = os.path.join(args.output_dir, "best")

    for epoch_idx in range(args.num_train_epochs):
        policy_model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        step_count = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            loss, accuracy = dpo_loss(policy_model, reference_model, batch, args.beta, device)
            (loss / args.gradient_accumulation_steps).backward()

            if step_idx % args.gradient_accumulation_steps == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_accuracy += accuracy
            step_count += 1

        train_metrics = {
            "loss": running_loss / max(1, step_count),
            "accuracy": running_accuracy / max(1, step_count),
        }
        print(
            f"Epoch {epoch_idx + 1}/{args.num_train_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )

        selection_metric = train_metrics["accuracy"]
        if val_loader is not None:
            val_metrics = evaluate(policy_model, reference_model, val_loader, args.beta, device)
            print(
                f"Epoch {epoch_idx + 1}/{args.num_train_epochs} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            selection_metric = val_metrics["accuracy"]

        if selection_metric > best_metric:
            best_metric = selection_metric
            os.makedirs(best_dir, exist_ok=True)
            policy_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved new best DPO checkpoint to: {best_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final DPO checkpoint to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a seq2seq model with DPO from saved preference pairs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the starting SFT checkpoint.")
    parser.add_argument("--reference_model_path", type=str, default=None, help="Optional reference checkpoint. Defaults to model_path.")
    parser.add_argument("--train_preferences", type=str, required=True, help="Path to training preference JSONL.")
    parser.add_argument("--validation_preferences", type=str, default=None, help="Optional path to validation preference JSONL.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save DPO checkpoints.")
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter.")
    parser.add_argument("--use_raw_prompt_only", action="store_true", help="Ignore prompt_with_prefix and use raw prompt field only.")
    args = parser.parse_args()
    main(args)

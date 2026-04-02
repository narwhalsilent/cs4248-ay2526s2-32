import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from rouge_score import rouge_scorer as rouge_scorer_lib
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    pipeline,
    AutoModelForCausalLM,
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


class AntiCopySeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, anti_copy_weight=0.0, pad_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anti_copy_weight = anti_copy_weight
        self.pad_token_id = pad_token_id

    def _compute_anti_copy_penalty(self, logits, labels, input_ids):
        if self.anti_copy_weight <= 0.0:
            return logits.new_zeros(())

        probs = F.softmax(logits.float(), dim=-1)
        batch_penalties = []

        for batch_idx in range(probs.size(0)):
            source_ids = input_ids[batch_idx]
            if self.pad_token_id is not None:
                source_ids = source_ids[source_ids != self.pad_token_id]
            source_ids = torch.unique(source_ids)
            if source_ids.numel() == 0:
                continue

            example_labels = labels[batch_idx]
            valid_positions = example_labels != -100
            if not torch.any(valid_positions):
                continue

            valid_label_ids = example_labels[valid_positions]
            gold_in_source = (valid_label_ids.unsqueeze(1) == source_ids.unsqueeze(0)).any(dim=1)
            penalized_positions = torch.where(valid_positions)[0][~gold_in_source]
            if penalized_positions.numel() == 0:
                continue

            source_copy_mass = probs[batch_idx, penalized_positions][:, source_ids].sum(dim=-1)
            batch_penalties.append(source_copy_mass.mean())

        if not batch_penalties:
            return logits.new_zeros(())

        return torch.stack(batch_penalties).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        outputs = model(**inputs)
        loss = outputs.loss

        if (
            self.anti_copy_weight > 0.0
            and labels is not None
            and input_ids is not None
            and hasattr(outputs, "logits")
        ):
            anti_copy_penalty = self._compute_anti_copy_penalty(
                outputs.logits,
                labels,
                input_ids,
            )
            loss = loss + self.anti_copy_weight * anti_copy_penalty

        return (loss, outputs) if return_outputs else loss


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_output_dir(base_output_dir, run_name=None, auto_suffix=False):
    if run_name:
        return f"{base_output_dir}_{run_name}"
    if auto_suffix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_output_dir}_{timestamp}"
    return base_output_dir


def build_inputs(factual_texts, source_prefix):
    if source_prefix:
        return [source_prefix + text for text in factual_texts]
    return factual_texts


def batched(iterable, batch_size):
    for start_idx in range(0, len(iterable), batch_size):
        yield start_idx, iterable[start_idx:start_idx + batch_size]


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize_text(text):
    return re.findall(r"[a-z']+", normalize_text(text))


def deduplicate_candidates(candidates):
    seen = set()
    deduped = []
    for candidate in candidates:
        key = normalize_text(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped or [""]


def factual_similarity(factual_text, generated_text):
    return SequenceMatcher(None, normalize_text(factual_text), normalize_text(generated_text)).ratio()


def lexical_overlap_fraction(source_text, candidate_text):
    source_tokens = set(tokenize_text(source_text))
    candidate_tokens = tokenize_text(candidate_text)
    if not source_tokens or not candidate_tokens:
        return 0.0
    overlap = sum(token in source_tokens for token in candidate_tokens)
    return overlap / len(candidate_tokens)


def longest_shared_ngram_ratio(source_text, candidate_text, max_n=4):
    source_tokens = tokenize_text(source_text)
    candidate_tokens = tokenize_text(candidate_text)
    if len(source_tokens) < 2 or len(candidate_tokens) < 2:
        return 0.0

    max_shared = 0
    max_considered_n = min(max_n, len(source_tokens), len(candidate_tokens))
    for n in range(2, max_considered_n + 1):
        source_ngrams = {
            tuple(source_tokens[idx:idx + n])
            for idx in range(len(source_tokens) - n + 1)
        }
        if not source_ngrams:
            continue
        for idx in range(len(candidate_tokens) - n + 1):
            if tuple(candidate_tokens[idx:idx + n]) in source_ngrams:
                max_shared = n
    return max_shared / max(1, min(len(source_tokens), len(candidate_tokens), max_n))


def copy_penalty(source_text, candidate_text):
    exact_copy = float(normalize_text(source_text) == normalize_text(candidate_text))
    sequence_similarity = factual_similarity(source_text, candidate_text)
    lexical_overlap = lexical_overlap_fraction(source_text, candidate_text)
    shared_ngram_ratio = longest_shared_ngram_ratio(source_text, candidate_text)
    return {
        "exact_copy": exact_copy,
        "sequence_similarity": sequence_similarity,
        "lexical_overlap": lexical_overlap,
        "shared_ngram_ratio": shared_ngram_ratio,
        "copy_penalty": (
            1.5 * exact_copy
            + 0.5 * sequence_similarity
            + 0.3 * lexical_overlap
            + 0.2 * shared_ngram_ratio
        ),
    }


def generate_candidate_predictions(
    model,
    tokenizer,
    factual_texts,
    source_prefix,
    device,
    batch_size,
    max_length,
    num_beams,
    num_candidates,
    candidate_strategy,
    temperature,
    top_p,
):
    model_inputs = build_inputs(factual_texts, source_prefix)
    all_candidates = []
    total_batches = (len(model_inputs) + batch_size - 1) // batch_size

    for _, batch in batched(model_inputs, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        generation_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_candidates,
        }
        if candidate_strategy == "sample":
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "num_beams": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "num_beams": max(num_beams, num_candidates),
                    "num_beam_groups": 1,
                }
            )

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for offset in range(0, len(decoded), num_candidates):
            candidates = [text.strip() for text in decoded[offset:offset + num_candidates]]
            all_candidates.append(deduplicate_candidates(candidates))

    if len(all_candidates) != len(factual_texts):
        raise ValueError("Candidate generation count mismatch.")

    return all_candidates


def calculate_text_perplexities(texts, model, tokenizer, batch_size):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for _, batch in batched(texts, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        token_losses = token_losses * shift_mask
        seq_losses = token_losses.sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
        perplexities.extend(torch.exp(seq_losses).cpu().tolist())

    return perplexities


def score_candidates(
    factual_texts,
    candidate_groups,
    sarcasm_clf,
    sbert_model,
    gpt2_model,
    gpt2_tokenizer,
    batch_size,
    style_weight,
    similarity_weight,
    copy_weight,
    fluency_weight,
):
    flat_candidates = []
    candidate_index = []
    for example_idx, candidates in enumerate(candidate_groups):
        for candidate in candidates:
            flat_candidates.append(candidate)
            candidate_index.append(example_idx)

    sarcasm_results = sarcasm_clf(flat_candidates, batch_size=batch_size)
    sarcasm_scores = [
        result["score"] if result["label"].lower() == "sarcastic" else 1 - result["score"]
        for result in sarcasm_results
    ]
    fluency_perplexities = calculate_text_perplexities(
        flat_candidates,
        gpt2_model,
        gpt2_tokenizer,
        batch_size=batch_size,
    )

    source_embeddings = sbert_model.encode(factual_texts, convert_to_tensor=True, show_progress_bar=True)
    candidate_embeddings = sbert_model.encode(flat_candidates, convert_to_tensor=True, show_progress_bar=True)

    grouped = [[] for _ in factual_texts]
    for flat_idx, candidate in enumerate(flat_candidates):
        example_idx = candidate_index[flat_idx]
        similarity_score = util.cos_sim(
            source_embeddings[example_idx].unsqueeze(0),
            candidate_embeddings[flat_idx].unsqueeze(0),
        ).item()
        copy_metrics = copy_penalty(factual_texts[example_idx], candidate)
        fluency_penalty = float(np.log(max(fluency_perplexities[flat_idx], 1e-8)))
        total_score = (
            style_weight * sarcasm_scores[flat_idx]
            + similarity_weight * similarity_score
            - copy_weight * copy_metrics["copy_penalty"]
            - fluency_weight * fluency_penalty
        )
        grouped[example_idx].append(
            {
                "candidate_text": candidate,
                "sarcasm_score": sarcasm_scores[flat_idx],
                "semantic_similarity": similarity_score,
                "fluency_perplexity": fluency_perplexities[flat_idx],
                "fluency_penalty": fluency_penalty,
                **copy_metrics,
                "rerank_score": total_score,
            }
        )

    return grouped


def build_preference_pairs(
    factual_texts,
    reference_texts,
    candidate_groups,
    source_prefix,
    pair_mode,
    min_score_margin,
):
    preference_pairs = []

    for example_idx, candidates in enumerate(candidate_groups):
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        if len(ranked) < 2:
            continue

        prompt_text = factual_texts[example_idx]
        prompt_with_prefix = (source_prefix or "") + prompt_text
        if pair_mode == "best_vs_worst":
            pair_indices = [(0, len(ranked) - 1)]
        else:
            pair_indices = [
                (chosen_idx, rejected_idx)
                for chosen_idx in range(len(ranked))
                for rejected_idx in range(chosen_idx + 1, len(ranked))
            ]

        for chosen_idx, rejected_idx in pair_indices:
            chosen = ranked[chosen_idx]
            rejected = ranked[rejected_idx]
            if chosen["candidate_text"] == rejected["candidate_text"]:
                continue
            score_margin = chosen["rerank_score"] - rejected["rerank_score"]
            if score_margin < min_score_margin:
                continue
            preference_pairs.append(
                {
                    "example_idx": example_idx,
                    "prompt": prompt_text,
                    "prompt_with_prefix": prompt_with_prefix,
                    "reference_satirical": reference_texts[example_idx],
                    "chosen": chosen["candidate_text"],
                    "rejected": rejected["candidate_text"],
                    "score_margin": score_margin,
                    "chosen_score": chosen["rerank_score"],
                    "rejected_score": rejected["rerank_score"],
                    "chosen_sarcasm": chosen["sarcasm_score"],
                    "rejected_sarcasm": rejected["sarcasm_score"],
                    "chosen_similarity": chosen["semantic_similarity"],
                    "rejected_similarity": rejected["semantic_similarity"],
                    "chosen_copy_penalty": chosen["copy_penalty"],
                    "rejected_copy_penalty": rejected["copy_penalty"],
                    "pair_mode": pair_mode,
                }
            )

    return preference_pairs


def save_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def export_preference_pairs(
    model,
    tokenizer,
    datasets_by_split,
    source_prefix,
    device,
    args,
    output_dir,
):
    print("\nLoading preference evaluators...")
    sarcasm_clf = pipeline(
        "text-classification",
        model="helinivan/english-sarcasm-detector",
        device=0 if device == "cuda" else -1
    )
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    requested_splits = [split.strip() for split in args.preference_splits.split(",") if split.strip()]
    for split_name in requested_splits:
        if split_name not in datasets_by_split:
            raise ValueError(f"Unknown preference split: {split_name}")

        split_df = datasets_by_split[split_name]
        factual_texts = split_df["factual"].astype(str).tolist()
        reference_texts = split_df["satirical"].astype(str).tolist()

        print(f"\nGenerating preference data for split: {split_name} ({len(split_df)} examples)")
        candidate_groups = generate_candidate_predictions(
            model=model,
            tokenizer=tokenizer,
            factual_texts=factual_texts,
            source_prefix=source_prefix,
            device=device,
            batch_size=args.preference_batch_size,
            max_length=args.preference_max_length,
            num_beams=args.preference_num_beams,
            num_candidates=args.preference_num_candidates,
            candidate_strategy=args.preference_candidate_strategy,
            temperature=args.preference_temperature,
            top_p=args.preference_top_p,
        )
        scored_candidates = score_candidates(
            factual_texts=factual_texts,
            candidate_groups=candidate_groups,
            sarcasm_clf=sarcasm_clf,
            sbert_model=sbert_model,
            gpt2_model=gpt2_model,
            gpt2_tokenizer=gpt2_tokenizer,
            batch_size=args.preference_batch_size,
            style_weight=args.preference_style_weight,
            similarity_weight=args.preference_similarity_weight,
            copy_weight=args.preference_copy_weight,
            fluency_weight=args.preference_fluency_weight,
        )
        preference_pairs = build_preference_pairs(
            factual_texts=factual_texts,
            reference_texts=reference_texts,
            candidate_groups=scored_candidates,
            source_prefix=source_prefix,
            pair_mode=args.preference_pair_mode,
            min_score_margin=args.preference_min_margin,
        )

        split_output_path = os.path.join(output_dir, f"{split_name}_preference_pairs.jsonl")
        save_jsonl(preference_pairs, split_output_path)
        print(f"Saved {len(preference_pairs)} preference pairs to: {split_output_path}")


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

    output_dir = resolve_output_dir(
        train_cfg["output_dir"],
        run_name=args.run_name,
        auto_suffix=args.auto_run_name,
    )
    anti_copy_weight = float(train_cfg.get("anti_copy_weight", 0.0))

    print(f"Model     : {model_name}")
    print(f"Type      : {model_type}")
    print(f"Output dir: {output_dir}")
    print(f"Anti-copy : {anti_copy_weight}")
    print(f"Device    : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (no GPU detected)'}")

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. Load Silver Dataset CSVs directly via pandas 
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

        # Use the modern target-tokenization API supported by current transformers.
        labels = tokenizer(
            text_target=examples["satirical"],
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

        # Newer trainer versions may return a tuple, and some evaluation paths hand
        # logits instead of token ids. Normalize to integer token ids before decode.
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        preds = np.nan_to_num(preds, nan=tokenizer.pad_token_id, posinf=tokenizer.pad_token_id, neginf=tokenizer.pad_token_id)
        preds = preds.astype(np.int64)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        if getattr(tokenizer, "vocab_size", None) is not None:
            preds = np.minimum(preds, tokenizer.vocab_size - 1)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels (padding) before decoding
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.nan_to_num(labels, nan=tokenizer.pad_token_id, posinf=tokenizer.pad_token_id, neginf=tokenizer.pad_token_id)
        labels = labels.astype(np.int64)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        if getattr(tokenizer, "vocab_size", None) is not None:
            labels = np.minimum(labels, tokenizer.vocab_size - 1)
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
    num_epochs = train_cfg["num_train_epochs"]
    batch_size = train_cfg["per_device_train_batch_size"]
    total_steps = (len(tokenized_datasets["train"]) // batch_size) * num_epochs
    warmup_steps = train_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg["weight_decay"],
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
        warmup_steps=warmup_steps,
        report_to=train_cfg.get("report_to", "none")
    )

    # 8. Initialize Trainer
    gpu_eta = total_steps * 0.7  # ~0.7s per step on GPU
    cpu_eta = total_steps * 8.0  # ~8s per step on CPU
    print(f"\nTraining estimate: {total_steps} steps total ({num_epochs} epochs x {total_steps // num_epochs} steps/epoch)")
    if torch.cuda.is_available():
        print(f"Estimated time  : ~{time.strftime('%H:%M:%S', time.gmtime(gpu_eta))} on GPU")
    else:
        print(f"Estimated time  : ~{time.strftime('%H:%M:%S', time.gmtime(cpu_eta))} on CPU (consider using a GPU)")

    trainer = AntiCopySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        anti_copy_weight=anti_copy_weight,
        pad_token_id=tokenizer.pad_token_id,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressCallback(total_epochs=num_epochs)
        ]
    )

    # 9. Train
    print("\nStarting training...")
    trainer.train()

    # 10. Save best model
    final_output = os.path.join(output_dir, "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\nBest model saved to: {final_output}")

    if args.export_preferences:
        preference_output_dir = args.preference_output_dir or os.path.join(output_dir, "preferences")
        export_preference_pairs(
            model=trainer.model,
            tokenizer=tokenizer,
            datasets_by_split={
                "train": train_df,
                "validation": val_df,
            },
            source_prefix=source_prefix,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args,
            output_dir=preference_output_dir,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART or T5 on the silver satire dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bart_training.yaml",
        help="Path to training config YAML (bart_training.yaml or t5_training.yaml)."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional suffix appended to the configured output_dir."
    )
    parser.add_argument(
        "--auto_run_name",
        action="store_true",
        help="Append a timestamp suffix to the configured output_dir."
    )
    parser.add_argument(
        "--export_preferences",
        action="store_true",
        help="After SFT finishes, generate preference pairs from the trained model."
    )
    parser.add_argument(
        "--preference_output_dir",
        type=str,
        default=None,
        help="Directory where train/validation preference JSONL files will be saved."
    )
    parser.add_argument(
        "--preference_splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits to export preference pairs for."
    )
    parser.add_argument(
        "--preference_batch_size",
        type=int,
        default=16,
        help="Batch size for preference candidate generation and scoring."
    )
    parser.add_argument(
        "--preference_max_length",
        type=int,
        default=128,
        help="Maximum decoding length for preference candidate generation."
    )
    parser.add_argument(
        "--preference_num_beams",
        type=int,
        default=4,
        help="Beam width when preference_candidate_strategy=beam."
    )
    parser.add_argument(
        "--preference_num_candidates",
        type=int,
        default=8,
        help="How many candidates to generate per example before building preference pairs."
    )
    parser.add_argument(
        "--preference_candidate_strategy",
        type=str,
        choices=["beam", "sample"],
        default="sample",
        help="How to generate multiple candidates for preference data."
    )
    parser.add_argument(
        "--preference_temperature",
        type=float,
        default=0.9,
        help="Sampling temperature when preference_candidate_strategy=sample."
    )
    parser.add_argument(
        "--preference_top_p",
        type=float,
        default=0.95,
        help="Top-p threshold when preference_candidate_strategy=sample."
    )
    parser.add_argument(
        "--preference_style_weight",
        type=float,
        default=2.0,
        help="Reranking weight for sarcasm score."
    )
    parser.add_argument(
        "--preference_similarity_weight",
        type=float,
        default=1.0,
        help="Reranking weight for semantic similarity."
    )
    parser.add_argument(
        "--preference_copy_weight",
        type=float,
        default=1.5,
        help="Penalty weight for copy-heavy candidates."
    )
    parser.add_argument(
        "--preference_fluency_weight",
        type=float,
        default=0.15,
        help="Penalty weight for GPT-2 log-perplexity during preference reranking."
    )
    parser.add_argument(
        "--preference_pair_mode",
        type=str,
        choices=["best_vs_worst", "all_pairs"],
        default="best_vs_worst",
        help="Whether to keep one pair per example or all valid ordered pairs."
    )
    parser.add_argument(
        "--preference_min_margin",
        type=float,
        default=0.05,
        help="Minimum reranker score gap required before saving a preference pair."
    )
    args = parser.parse_args()
    if args.export_preferences and args.preference_num_candidates <= 1:
        parser.error("--export_preferences requires --preference_num_candidates > 1.")
    main(args)

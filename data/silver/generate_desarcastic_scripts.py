import argparse
import json
import logging
import os
import re
import warnings
from pathlib import Path

from openai import OpenAI
import torch
from bert_score import scorer
from tqdm import tqdm
from transformers import pipeline


warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = str((SCRIPT_DIR.parent / "raw" / "Sarcasm_Headlines_Dataset.json").resolve())
DEFAULT_OUTPUT_FILE = str((SCRIPT_DIR / "desarcastic_headlines_1788.txt").resolve())
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_CANDIDATES = 5
DEFAULT_BATCH_SIZE = 6
DEFAULT_MAX_NEW_TOKENS = 160


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_name}")

pipeline_kwargs = {
    "task": "text-classification",
    "model": "helinivan/english-sarcasm-detector",
    "device": device_name,
}
if torch.cuda.is_available():
    pipeline_kwargs["dtype"] = torch.float16

sarcasm_pipe = pipeline(**pipeline_kwargs)

bert_scorer = scorer.BERTScorer(
    lang="en",
    rescale_with_baseline=True,
    device=device_name,
)

logger = logging.getLogger("desarcastic_generation")


def setup_logger(_log_path):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def build_openrouter_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Export it before running this script."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def extract_json_object(text):
    if not text:
        return None

    cleaned = text.strip()
    code_fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_fence_match:
        cleaned = code_fence_match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    return None


def extract_headlines_from_output(text):
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        headlines = parsed.get("headlines", [])
        if isinstance(headlines, list):
            return [str(item).strip() for item in headlines if str(item).strip()]
    return []


def build_messages(headline, num_candidates):
    system_prompt = (
        "You rewrite satirical headlines into straight, factual headlines.\n"
        f"Generate exactly {num_candidates} distinct non-sarcastic headlines.\n"
        "Preserve the core event, entities, and meaning from the source.\n"
        "Remove irony, exaggeration, mockery, and punchline wording.\n"
        "Keep each headline concise, news-style, and semantically clear.\n"
        "OUTPUT FORMAT: Respond ONLY with a JSON object. No prose.\n"
        "Example: {\"headlines\": [\"H1\", \"H2\", \"H3\", \"H4\", \"H5\"]}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Satirical Headline: {headline}"},
    ]


def call_teacher_llm(headline, client, model_name, num_candidates, max_new_tokens):
    messages = build_messages(headline, num_candidates)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://github.com/choongkaizhe/cs4248-ay2526s2-32",
                "X-Title": "cs4248 desarcastic generation",
            },
        )
        raw_text = response.choices[0].message.content.strip() if response.choices else ""
        headlines = extract_headlines_from_output(raw_text)

        if not headlines:
            logger.warning("Teacher output was not parseable JSON for headline: %s", headline)
            logger.warning("Raw teacher output: %s", raw_text[:500])
            tqdm.write(f"No candidates returned for headline: {headline}")
            return []

        deduped = []
        seen = set()
        for candidate in headlines:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)

        return deduped[:num_candidates]
    except Exception:
        tqdm.write(f"LLM generation failed for headline: {headline}")
        logger.exception("LLM generation failed for headline: %s", headline)
        return []


def process_batch(batch, sarcasm_classifier, bertscore_scorer, out_f):
    if not batch:
        return

    all_candidates = []
    item_lengths = []
    for item in batch:
        all_candidates.extend(item["cands"])
        item_lengths.append(len(item["cands"]))

    try:
        with torch.inference_mode():
            sarcasm_results = sarcasm_classifier(
                all_candidates,
                batch_size=len(all_candidates),
            )
            non_sarcasm_scores = [
                result["score"] if result["label"] == "LABEL_0" else 1 - result["score"]
                for result in sarcasm_results
            ]

            references = []
            for item in batch:
                references.extend([item["original"]] * len(item["cands"]))

            _, _, f1_tensor = bertscore_scorer.score(all_candidates, references)
            content_scores = f1_tensor.tolist()
    except Exception:
        originals = [item["original"] for item in batch]
        logger.exception("Batch scoring failed for headlines: %s", originals)
        tqdm.write("Batch scoring failed. Check Slurm logs for details.")
        return

    current_idx = 0
    for batch_idx, item in enumerate(batch):
        num_cands = item_lengths[batch_idx]
        item_cands = item["cands"]
        item_non_sarcasm = non_sarcasm_scores[current_idx : current_idx + num_cands]
        item_content = content_scores[current_idx : current_idx + num_cands]
        current_idx += num_cands

        scored_candidates = []
        for cand_idx, candidate in enumerate(item_cands):
            total = (item_non_sarcasm[cand_idx] + item_content[cand_idx]) / 2
            scored_candidates.append(
                {
                    "headline": candidate,
                    "non_sarcasm_score": item_non_sarcasm[cand_idx],
                    "content_score": item_content[cand_idx],
                    "total": total,
                }
            )

        scored_candidates.sort(key=lambda item_: item_["total"], reverse=True)
        winner = scored_candidates[0]

        tqdm.write(f"\n[SATIRICAL]: {item['original']}")
        tqdm.write(
            " ★ WINNER: "
            f"{winner['headline']} "
            f"(Non-sarcasm: {winner['non_sarcasm_score']:.3f}, "
            f"BERTScore: {winner['content_score']:.3f}, "
            f"Total: {winner['total']:.3f})"
        )

        result = {
            "original_sarcastic_headline": item["original"],
            "silver_factual_headline": winner["headline"],
            "non_sarcasm_score": round(winner["non_sarcasm_score"], 4),
            "content_score": round(winner["content_score"], 4),
            "confidence_score": round(winner["total"], 4),
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()


def load_existing_headlines(output_path):
    output_path = Path(output_path)
    processed_headlines = set()
    if not output_path.exists():
        return processed_headlines

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            headline = record.get("original_sarcastic_headline")
            if headline:
                processed_headlines.add(headline)

    return processed_headlines


def load_source_headlines(input_path, processed_headlines, limit=None):
    input_path = Path(input_path)
    to_process = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in input: %s", line[:200])
                continue

            if item.get("is_sarcastic") != 1:
                continue

            headline = item.get("headline")
            if not headline or headline in processed_headlines:
                continue

            to_process.append(item)
            if limit and len(to_process) >= limit:
                break

    return to_process


def generate_desarcastic_dataset(
    input_path,
    output_path,
    limit=None,
    model_name=DEFAULT_MODEL,
    num_candidates=DEFAULT_CANDIDATES,
    batch_size=DEFAULT_BATCH_SIZE,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
):
    setup_logger(None)
    output_path = str(Path(output_path).resolve())
    input_path = str(Path(input_path).resolve())
    logger.info(
        "Starting desarcastic generation | input=%s | output=%s | model=%s | "
        "num_candidates=%s | batch_size=%s | max_new_tokens=%s",
        input_path,
        output_path,
        model_name,
        num_candidates,
        batch_size,
        max_new_tokens,
    )

    client = build_openrouter_client()

    processed_headlines = load_existing_headlines(output_path)
    remaining_limit = None
    if limit is not None:
        remaining_limit = max(limit - len(processed_headlines), 0)

    to_process = load_source_headlines(input_path, processed_headlines, remaining_limit)
    print(
        f"Resuming: {len(processed_headlines)} headlines done. "
        f"Starting {len(to_process)} sarcastic headlines..."
    )

    batch_buffer = []
    with open(output_path, "a", encoding="utf-8") as out_f:
        for item in tqdm(to_process):
            original_headline = item["headline"]
            candidates = call_teacher_llm(
                original_headline,
                client,
                model_name,
                num_candidates,
                max_new_tokens,
            )

            if candidates:
                batch_buffer.append({"original": original_headline, "cands": candidates})
            else:
                logger.warning("No usable candidates for headline: %s", original_headline)

            if len(batch_buffer) >= batch_size:
                process_batch(batch_buffer, sarcasm_pipe, bert_scorer, out_f)
                batch_buffer = []

        if batch_buffer:
            process_batch(batch_buffer, sarcasm_pipe, bert_scorer, out_f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate desatirised silver headlines from sarcastic headlines."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_desarcastic_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        model_name=args.model,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

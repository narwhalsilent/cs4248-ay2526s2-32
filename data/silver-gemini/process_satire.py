import asyncio
import json
import os
import time
import re
from google import genai

# --- CONFIGURATION ---
MODEL_NAME  = "models/gemma-3-12b-it"
API_KEY     = "AIzaSyBJwYsGJqd5yb6iTq2ubxT2K-2Hxm9Y1FA"
INPUT_PATH  = "../raw/Sarcasm_Headlines_Dataset_v2.json"
OUTPUT_PATH = "./processed_headlines.json"

# ---------------------------------------------------------------------------
# RATE LIMITS — Gemma free tier
#   RPD : 14,400 | RPM : 30 | TPM : 15,000
#
# Per-batch estimate (BATCH_SIZE=10):
#   Input  ~620 tokens | Output ~750 tokens | Total ~1,370 tokens
#
# Binding constraint = TPM: 15,000 / 1,370 ≈ 10.9 batches/min max.
# We use 3 workers (down from 5) to reduce 503 pressure on the model.
# ---------------------------------------------------------------------------
BATCH_SIZE     = 10
MAX_CONCURRENT = 3      # Reduced from 5 — fewer concurrent hits = fewer 503s
RPM_LIMIT      = 20     # Reduced from 25 — more headroom under 30 RPM cap
TPM_LIMIT      = 13_000

INPUT_COST_PER_TOKEN  = 0.0 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.0 / 1_000_000
EST_INPUT_TOKENS  = 620
EST_OUTPUT_TOKENS = 750

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------------
# ISSUE 1 FIX: Pragmatic Metacognitive Prompt (PMP)
#
# Key principles applied:
#   - Role priming: forces the model into the exact satirist persona
#   - Metacognitive instruction: explicitly tells the model WHAT makes good
#     Onion-style sarcasm (specificity, deadpan tone, semantic retention)
#   - Negative examples: tells it what NOT to do (vague, generic reactions)
#   - Hard formatting contract: exact JSON schema with no wiggle room
#   - Chain-of-thought suppression: "Output ONLY the JSON" prevents prose leakage
# ---------------------------------------------------------------------------
PMP_SYSTEM_PROMPT = """You are a staff writer for The Onion, the world's leading satirical newspaper. Your craft is Onion-style deadpan satire: you take a real headline and rewrite it to expose its absurdity while keeping every specific detail intact.

WHAT MAKES A GREAT ONION HEADLINE:
1. SEMANTIC RETENTION — Every specific noun, person, place, number, and event from the original MUST appear or be directly referenced. Never replace specifics with vague phrases.
2. DEADPAN DELIVERY — State the absurd as if it is completely normal. No exclamation points. No "Oh joy" or "Because we clearly need...". Treat the ridiculous with bureaucratic seriousness.
3. SPECIFICITY — The more specific the better. "Nation's Muslims Await Registry Form 27-B" beats "Government Considering Tracking People".
4. IRONY THROUGH IMPLICATION — Let the reader connect the dots. Do not explain the joke.

BAD EXAMPLE (what NOT to do):
Input: "trump team mulling muslim registry"
Bad candidate: "Just casually considering a Muslim registry, no big deal." ← vague, sarcasm is stated not implied, loses all specifics

GOOD EXAMPLE (what to do):
Input: "trump team mulling muslim registry"  
Good candidate: "Trump Transition Team Assures Nation Muslim Registry Will Be 'Very Tasteful, Very Professional'" ← keeps all specifics, deadpan, absurdity through bureaucratic framing

YOUR TASK: For each factual headline in the DATA array, generate exactly 5 Onion-style sarcastic headline variations following the above principles.

OUTPUT RULES — FOLLOW EXACTLY:
- Output ONLY a raw JSON array. No prose before or after. No markdown. No backticks.
- Each object must have exactly these keys: factual_headline, candidate_1, candidate_2, candidate_3, candidate_4, candidate_5
- All strings must use only standard ASCII or escaped unicode. No em-dashes, smart quotes, or ellipsis characters.
- Schema: [{"factual_headline": "...", "candidate_1": "...", "candidate_2": "...", "candidate_3": "...", "candidate_4": "...", "candidate_5": "..."}]"""


# ---------------------------------------------------------------------------
# ISSUE 2 FIX: Unicode normalisation
# \u2026 = ellipsis, \u2013 = en-dash, \u2014 = em-dash, \u2018/\u2019 = smart quotes
# We decode these to their ASCII equivalents after parsing so the JSON file
# stays clean and human-readable.
# ---------------------------------------------------------------------------
UNICODE_REPLACEMENTS = {
    "\u2026": "...",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00e9": "e",
    "\u00e8": "e",
    "\u00ea": "e",
    "\u00e0": "a",
    "\u00e2": "a",
    "\u00f4": "o",
    "\u00fb": "u",
    "\u00e7": "c",
    "\u00f1": "n",
    "\u00fc": "u",
    "\u00e4": "a",
    "\u00f6": "o",
    "\u00df": "ss",
}

def clean_unicode(text: str) -> str:
    for char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text

def clean_record(record: dict) -> dict:
    return {k: clean_unicode(v) if isinstance(v, str) else v for k, v in record.items()}


# ---------------------------------------------------------------------------
# RATE LIMITER (RPM + TPM dual guard)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.min_interval   = 60.0 / rpm
        self.tpm_limit      = tpm
        self.rpm_lock       = asyncio.Lock()
        self.tpm_lock       = asyncio.Lock()
        self.last_call_time = 0.0
        self.token_log: list[tuple[float, int]] = []

    async def acquire_rpm(self):
        async with self.rpm_lock:
            now  = time.monotonic()
            wait = self.min_interval - (now - self.last_call_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call_time = time.monotonic()

    async def record_tokens(self, tokens_used: int):
        async with self.tpm_lock:
            now = time.monotonic()
            self.token_log = [(t, n) for t, n in self.token_log if now - t < 60.0]
            tokens_last_min = sum(n for _, n in self.token_log)
            if tokens_last_min + tokens_used > self.tpm_limit:
                oldest    = self.token_log[0][0] if self.token_log else now
                sleep_for = 60.0 - (now - oldest) + 1.0
                print(f"  [TPM ] Token cap approaching — sleeping {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                self.token_log = [(t, n) for t, n in self.token_log if now - t < 60.0]
            self.token_log.append((time.monotonic(), tokens_used))


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def extract_json(text: str) -> list:
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


# ---------------------------------------------------------------------------
# BATCH PROCESSOR
# ISSUE 3 FIX: 503 errors now retry indefinitely with escalating backoff
# instead of giving up after 3 attempts. 429s also retry indefinitely.
# Only genuine parse/logic errors are limited to 3 attempts.
# ---------------------------------------------------------------------------
async def process_batch(
    batch_headlines : list,
    semaphore       : asyncio.Semaphore,
    rate_limiter    : RateLimiter,
    batch_index     : int,
    total_batches   : int,
    output_file,
    write_lock      : asyncio.Lock,
) -> int:

    async with semaphore:
        await rate_limiter.acquire_rpm()

        prompt = (
            PMP_SYSTEM_PROMPT +
            f"\n\nDATA: {json.dumps(batch_headlines)}"
        )

        parse_failures = 0
        attempt        = 0

        while True:
            attempt     += 1
            batch_start  = time.monotonic()
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model    = MODEL_NAME,
                    contents = prompt,
                )

                elapsed = time.monotonic() - batch_start
                result  = extract_json(response.text)

                # Clean unicode from every record
                result = [clean_record(r) for r in result]

                in_tok  = (response.usage_metadata.prompt_token_count  or EST_INPUT_TOKENS)
                out_tok = (response.usage_metadata.candidates_token_count or EST_OUTPUT_TOKENS)
                total_tokens = in_tok + out_tok
                cost = (in_tok * INPUT_COST_PER_TOKEN + out_tok * OUTPUT_COST_PER_TOKEN)

                await rate_limiter.record_tokens(total_tokens)

                async with write_lock:
                    for record in result:
                        output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    output_file.flush()

                print(
                    f"  [OK   {batch_index:>4}/{total_batches}] "
                    f"{len(result):>2} records | "
                    f"{in_tok}in + {out_tok}out = {total_tokens} tok | "
                    f"cost: ${cost:.6f} | {elapsed:.2f}s"
                )
                return len(result)

            except Exception as e:
                elapsed = time.monotonic() - batch_start
                err_str = str(e)

                if "429" in err_str:
                    # Rate limited — parse suggested delay, retry indefinitely
                    delay_match = re.search(r'retry[^\d]*(\d+)', err_str, re.IGNORECASE)
                    sleep_for   = int(delay_match.group(1)) + 5 if delay_match else 65
                    print(f"  [429  {batch_index:>4}/{total_batches}] "
                          f"Rate limited — sleeping {sleep_for}s (attempt {attempt})")
                    await asyncio.sleep(sleep_for)

                elif "503" in err_str or "UNAVAILABLE" in err_str:
                    # Model overloaded — retry indefinitely with escalating backoff
                    # Cap at 120s so we don't wait forever between attempts
                    sleep_for = min(15 * attempt, 120)
                    print(f"  [503  {batch_index:>4}/{total_batches}] "
                          f"Model unavailable — sleeping {sleep_for}s (attempt {attempt})")
                    await asyncio.sleep(sleep_for)

                else:
                    # Genuine error (JSON parse, network, etc.) — limit to 3 attempts
                    parse_failures += 1
                    print(f"  [WARN {batch_index:>4}/{total_batches}] "
                          f"Attempt {parse_failures}/3 failed after {elapsed:.2f}s: {e}")
                    if parse_failures >= 3:
                        print(f"  [FAIL {batch_index:>4}/{total_batches}] "
                              f"3 parse/logic failures. Skipping batch.")
                        return 0
                    await asyncio.sleep(5 * parse_failures)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}")
        return

    all_factual = []
    with open(INPUT_PATH, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("is_sarcastic") == 0:
                    all_factual.append(data["headline"])
            except Exception:
                continue

    processed = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["factual_headline"])
                except Exception:
                    continue

    to_process    = [h for h in all_factual if h not in processed]
    total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total headlines : {len(all_factual)}")
    print(f"Already done    : {len(processed)}")
    print(f"Remaining       : {len(to_process)}  ({total_batches} batches)")
    print(f"Concurrency     : {MAX_CONCURRENT} workers | RPM: {RPM_LIMIT} | TPM: {TPM_LIMIT}\n")

    if not to_process:
        print("Nothing left — all done!")
        return

    semaphore    = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = RateLimiter(rpm=RPM_LIMIT, tpm=TPM_LIMIT)
    write_lock   = asyncio.Lock()
    run_start    = time.monotonic()

    with open(OUTPUT_PATH, 'a') as output_file:
        tasks = [
            process_batch(
                batch_headlines = to_process[i : i + BATCH_SIZE],
                semaphore       = semaphore,
                rate_limiter    = rate_limiter,
                batch_index     = (i // BATCH_SIZE) + 1,
                total_batches   = total_batches,
                output_file     = output_file,
                write_lock      = write_lock,
            )
            for i in range(0, len(to_process), BATCH_SIZE)
        ]
        counts = await asyncio.gather(*tasks)

    total_time = time.monotonic() - run_start
    print(f"\nDone — {sum(counts)} new records written to {OUTPUT_PATH}")
    print(f"Run time: {total_time:.1f}s  ({total_time / 60:.1f} min)")

if __name__ == "__main__":
    asyncio.run(main())
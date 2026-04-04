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
OUTPUT_PATH = "./processed_headlines_desatirized.json"

# ---------------------------------------------------------------------------
# RATE LIMITS — Gemma free tier
#   RPD : 14,400 | RPM : 30 | TPM : 15,000
# ---------------------------------------------------------------------------
BATCH_SIZE     = 10
MAX_CONCURRENT = 3
RPM_LIMIT      = 20
TPM_LIMIT      = 13_000

INPUT_COST_PER_TOKEN  = 0.0 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.0 / 1_000_000
EST_INPUT_TOKENS  = 620
EST_OUTPUT_TOKENS = 750

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------------
# PMP PROMPT — Desatirization direction
#
# Role: AP/Reuters wire journalist, not a satirist.
# Task: Take an Onion or HuffPost satirical headline and rewrite it as a
#       straight, neutral, factual news headline — stripping all irony,
#       deadpan humour, and exaggeration while preserving the real-world
#       subject matter underneath.
# ---------------------------------------------------------------------------
PMP_SYSTEM_PROMPT = """You are a senior wire journalist at the Associated Press. Your job is to receive satirical or sensationalist headlines — written in the style of The Onion or HuffPost — and rewrite them as neutral, factual, straight-news headlines that could appear on Reuters or the BBC.

WHAT MAKES A GREAT DESATIRIZED HEADLINE:
1. SEMANTIC RETENTION — The real-world subject, person, place, event, or policy from the satirical headline MUST be preserved. Never discard the core topic.
2. NEUTRALITY — Remove all irony, sarcasm, deadpan humour, exaggeration, and loaded language. Report the underlying fact plainly.
3. SPECIFICITY — Keep all specific nouns, names, numbers, and events. Replace vague satirical framing with precise factual language.
4. WIRE STYLE — Short, direct, present-tense or past-tense declarative sentence. No opinion. No jokes. No rhetorical questions.

BAD EXAMPLE (what NOT to do):
Input: "Trump Transition Team Assures Nation Muslim Registry Will Be 'Very Tasteful, Very Professional'"
Bad candidate: "Government considers tracking religious groups." <- loses all specifics, still vague

GOOD EXAMPLE (what to do):
Input: "Trump Transition Team Assures Nation Muslim Registry Will Be 'Very Tasteful, Very Professional'"
Good candidate: "Trump advisers discuss proposal to create registry tracking Muslim immigrants" <- specific, neutral, factual, no humour

YOUR TASK: For each satirical headline in the DATA array, generate exactly 5 distinct desatirized factual headline variations following the above principles.

OUTPUT RULES — FOLLOW EXACTLY:
- Output ONLY a raw JSON array. No prose before or after. No markdown. No backticks.
- Each object must have exactly these keys: satirical_headline, candidate_1, candidate_2, candidate_3, candidate_4, candidate_5
- All strings must use only standard ASCII. No em-dashes, smart quotes, or ellipsis characters.
- Schema: [{"satirical_headline": "...", "candidate_1": "...", "candidate_2": "...", "candidate_3": "...", "candidate_4": "...", "candidate_5": "..."}]"""


# ---------------------------------------------------------------------------
# Unicode normalisation
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
    """
    Extract JSON array from response, with salvage mode for truncated output.
    """
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Salvage mode — recover individual complete objects from a truncated array
    salvaged = []
    for obj_match in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            salvaged.append(json.loads(obj_match.group(0)))
        except json.JSONDecodeError:
            continue

    if salvaged:
        print(f"      [SALVAGE] Truncated response — recovered {len(salvaged)} complete objects.")
        return salvaged

    raise ValueError(f"Could not extract any JSON from response. Preview: {text[:120]}")


# ---------------------------------------------------------------------------
# BATCH PROCESSOR
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
                    config   = {"max_output_tokens": 8192},
                )

                elapsed = time.monotonic() - batch_start
                result  = extract_json(response.text)
                result  = [clean_record(r) for r in result]

                in_tok  = (response.usage_metadata.prompt_token_count      or EST_INPUT_TOKENS)
                out_tok = (response.usage_metadata.candidates_token_count  or EST_OUTPUT_TOKENS)
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
                    delay_match = re.search(r'retry[^\d]*(\d+)', err_str, re.IGNORECASE)
                    sleep_for   = int(delay_match.group(1)) + 5 if delay_match else 65
                    print(f"  [429  {batch_index:>4}/{total_batches}] "
                          f"Rate limited — sleeping {sleep_for}s (attempt {attempt})")
                    await asyncio.sleep(sleep_for)

                elif "503" in err_str or "UNAVAILABLE" in err_str:
                    sleep_for = min(15 * attempt, 120)
                    print(f"  [503  {batch_index:>4}/{total_batches}] "
                          f"Model unavailable — sleeping {sleep_for}s (attempt {attempt})")
                    await asyncio.sleep(sleep_for)

                else:
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

    # 1. Load all SATIRICAL headlines (is_sarcastic == 1)
    all_satirical = []
    with open(INPUT_PATH, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("is_sarcastic") == 1:
                    all_satirical.append(data["headline"])
            except Exception:
                continue

    # 2. Resume — skip headlines already written to output
    processed = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["satirical_headline"])
                except Exception:
                    continue

    to_process    = [h for h in all_satirical if h not in processed]
    total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total satirical headlines : {len(all_satirical)}")
    print(f"Already done              : {len(processed)}")
    print(f"Remaining                 : {len(to_process)}  ({total_batches} batches)")
    print(f"Concurrency               : {MAX_CONCURRENT} workers | RPM: {RPM_LIMIT} | TPM: {TPM_LIMIT}\n")

    if not to_process:
        print("Nothing left — all done!")
        return

    semaphore    = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = RateLimiter(rpm=RPM_LIMIT, tpm=TPM_LIMIT)
    write_lock   = asyncio.Lock()
    run_start    = time.monotonic()

    # 3. Append mode — previous records are never touched
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
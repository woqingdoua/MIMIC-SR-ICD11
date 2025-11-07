#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate patient self-reports with GPT-5 and write ONLY 5 fields back to the original CSV.
File: ehr_report_sample.csv  (must have: persona, text, report_gpt4o_mini)

Writes/updates columns:
  Faithfulness, ClinicalCorrectness, Persona, Realism, overall

Safeguards:
  - Checkpoint every --save_every rows to <output or input>.tmp
  - Final atomic replace; optional one-time .bak backup of original when overwriting input

Usage example:
  python eval_gpt5_min5cols.py --input ehr_report_sample.csv --model gpt-5 --save_every 100 --only_missing 1

Deps:
  pip install -U openai pandas httpx tenacity tqdm
Env:
  export OPENAI_API_KEY=...
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# -------------------- Prompt --------------------
EVAL_PROMPT = """You are a clinical writing evaluator. Use ONLY the persona and the PATIENT-STATED EHR excerpt as ground truth. Evaluate the candidate self-report strictly against that.

Inputs
- Persona (JSON): {persona_json}
- EHR (patient-stated only):
{ehr_text}
- Candidate self-report:
{candidate_report}

Scores (1–5) & weights
- ClinicalCorrectness (0.40): Score only the claims actually made in the report (affirmed symptoms or explicit denials).
  • Zero-claim rule: if the report makes NO symptom claims at all, set ClinicalCorrectness = 5.0.
  • Semantic match: count synonyms/abbreviations/paraphrases if they clearly refer to the same symptom TODAY (e.g., SOB ↔ shortness of breath; loose stools ↔ diarrhea; can’t keep food down ↔ vomiting).
  • Negation scope & timing: a denial is supported only if TODAY’S EHR states the same denial.
  • Contradictions: if the report asserts the opposite of TODAY’S EHR, penalize heavily.
  • Unsupported new facts: penalize new medical facts not in the EHR; do NOT penalize neutral paraphrases.
  • Omissions: do NOT penalize omissions unless they create a contradiction.
- Faithfulness (0.30): Only patient-stated facts; no leakage of exam, vitals interpretation, labs/imaging, inpatient course, diagnoses, or treatment speculation.
- Persona (0.20): Tone/word choice align with persona cues (education, age, language, setting) without adding facts.
- Realism (0.10): Natural first-person voice; concise; selective denials (not exhaustive lists); plausible length/cadence for the setting/persona.

overall = 0.30*Faithfulness + 0.40*ClinicalCorrectness + 0.20*Persona + 0.10*Realism

Output (JSON only)
{{
  "Faithfulness": <1-5>,
  "ClinicalCorrectness": <1-5>,
  "Persona": <1-5>,
  "Realism": <1-5>,
  "overall": <1-5>
}}
"""

TARGET_KEYS = ["Faithfulness", "ClinicalCorrectness", "Persona", "Realism", "overall"]

# -------------------- Helpers --------------------
def strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    return m.group(1) if m else text

def build_prompt(persona: str, ehr: str, report: str) -> str:
    persona_json = persona if isinstance(persona, str) else ""
    ehr_text = ehr if isinstance(ehr, str) else ""
    cand = report if isinstance(report, str) else ""
    return EVAL_PROMPT.format(persona_json=persona_json, ehr_text=ehr_text, candidate_report=cand)

def get_client(api_key: str, timeout: int = 60) -> OpenAI:
    proxy = os.environ.get("HTTPS_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=timeout) if proxy else httpx.Client(timeout=timeout)
    return OpenAI(api_key=api_key, http_client=http_client, timeout=timeout)

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def eval_once(client: OpenAI, model: str, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Prefer Responses API (max_output_tokens). Fallback to Chat Completions (no max_tokens).
    Return (parsed_dict or None, raw_text_or_error).
    """
    # Preferred: Responses API
    try:
        resp = client.responses.create(
            model=model,
            max_output_tokens=160,  # just JSON
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "Return JSON only, no extra text."}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ],
        )
        txt = getattr(resp, "output_text", None)
        if not txt:
            parts = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            parts.append(c.text or "")
            txt = "".join(parts)
        raw = strip_code_fences(txt or "")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data, raw
        return None, f"__ERROR__:NOT_DICT | RAW={raw[:500]}"
    except Exception:
        pass

    # Fallback: Chat Completions（不传 max_tokens）
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only, no extra text."},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = strip_code_fences(resp2.choices[0].message.content or "")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data, raw
        return None, f"__ERROR__:NOT_DICT | RAW={raw[:500]}"
    except Exception as e:
        return None, f"__ERROR__:JSON_PARSE:{str(e)[:200]} | RAW={raw[:500]}"

def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for k in TARGET_KEYS:
        if k not in df.columns:
            df[k] = pd.NA
    return df

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ehr_subset_10k_gpt5_nopersona.csv")
    ap.add_argument("--output", default="", help="final output (default: overwrite input)")
    ap.add_argument("--tmp", default="", help="checkpoint path (default: <output or input>.tmp)")
    ap.add_argument("--model", default="gpt-4.1", help="gpt-5 / gpt-5-mini / gpt-4o-mini ...")
    ap.add_argument("--api_key",
                    default="",
                    help="OpenAI API key (or set OPENAI_API_KEY).")

    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=100, help="checkpoint frequency (rows)")
    ap.add_argument("--only_missing", type=int, default=1, help="1=only rows with missing 'overall'")
    ap.add_argument("--backup", type=int, default=1, help="create one-time .bak before final replace when overwriting input")
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Export it or pass --api_key.")

    out_path = Path(args.output or args.input)
    tmp_path = Path(args.tmp or (str(out_path) + ".tmp"))

    # Load (resume if .tmp exists)
    if tmp_path.exists():
        df = pd.read_csv(tmp_path)
        print(f"[resume] Loaded checkpoint: {tmp_path} (rows={len(df)})")
    else:
        df = pd.read_csv(out_path)
        print(f"[start] Loaded input: {out_path} (rows={len(df)})")

    # Required input columns
    for col in ["persona", "text", "report"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Ensure metric columns (ONLY 5 fields)
    df = ensure_metric_columns(df)

    # Row selection
    idxs = list(range(len(df)))
    if args.limit > 0:
        idxs = idxs[: args.limit]
    if args.only_missing:
        idxs = [i for i in idxs if pd.isna(df.at[i, "overall"])]

    if not idxs:
        # finalize immediately
        df.to_csv(tmp_path, index=False)
        if args.backup and (args.output == "" or out_path == Path(args.input)) and Path(args.input).exists():
            bak_path = Path(args.input).with_suffix(Path(args.input).suffix + ".bak")
            pd.read_csv(args.input).to_csv(bak_path, index=False)
            print(f"[backup] Saved backup: {bak_path}")
        os.replace(tmp_path, out_path)
        print("[finalize] No rows to process.")
        return

    client = get_client(args.api_key, args.timeout)
    save_every = max(1, int(args.save_every))

    processed = 0
    errors = 0
    pbar = tqdm(total=len(idxs), desc=f"Scoring → {args.model}")

    for i in idxs:
        persona = str(df.at[i, "persona"])
        ehr = str(df.at[i, "text"])
        rep = str(df.at[i, "report"])  # 被评估对象

        prompt = build_prompt(persona, ehr, rep)

        try:
            data, raw = eval_once(client, args.model, prompt)
            if data is None:
                errors += 1
            else:
                # 仅写 5 个评分字段
                for k in TARGET_KEYS:
                    if k in data:
                        df.at[i, k] = data[k]
        except Exception as e:
            errors += 1

        processed += 1
        pbar.update(1)

        if processed % save_every == 0:
            df.to_csv(tmp_path, index=False)
            print(f"[checkpoint] Saved {processed} rows → {tmp_path} (errors: {errors})")

    pbar.close()

    # final checkpoint
    df.to_csv(tmp_path, index=False)
    print(f"[checkpoint] Final checkpoint → {tmp_path}")

    # one-time .bak of the original (only when overwriting input)
    if args.backup and (args.output == "" or out_path == Path(args.input)) and Path(args.input).exists():
        bak_path = Path(args.input).with_suffix(Path(args.input).suffix + ".bak")
        pd.read_csv(args.input).to_csv(bak_path, index=False)
        print(f"[backup] Saved backup: {bak_path}")

    # atomic replace
    os.replace(tmp_path, out_path)
    print(f"[done] Wrote final to: {out_path} | processed={processed}, errors={errors}")

if __name__ == "__main__":
    main()

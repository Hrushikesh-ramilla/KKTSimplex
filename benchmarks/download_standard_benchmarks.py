"""
Downloader for official standard sycophancy & reasoning benchmarks:
1. Anthropic / Sharma et al. (2023): 'are_you_sure.jsonl' (HuggingFace: meg-tong/sycophancy-eval)
2. Anthropic / Perez et al. (2022): 'sycophancy_on_nlp_survey.jsonl' (HuggingFace: Anthropic/model-written-evals)
3. OpenAI GSM8K (Cobbe et al. 2021): Sourced from local benchmark assets
"""

import json
import os
import shutil
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url: str, dest_filename: str):
    dest_path = os.path.join(DATA_DIR, dest_filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[CACHE] {dest_filename} already exists ({os.path.getsize(dest_path)} bytes).")
        return dest_path
    
    print(f"[DOWNLOADING] {dest_filename} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"[DONE] Saved {dest_filename} ({os.path.getsize(dest_path)} bytes).")
    return dest_path


def main():
    # 1. Download Sharma et al. (2023) / Anthropic 'are_you_sure' & 'feedback' sycophancy benchmark
    url_are_you_sure = "https://huggingface.co/datasets/meg-tong/sycophancy-eval/raw/main/are_you_sure.jsonl"
    download_file(url_are_you_sure, "are_you_sure.jsonl")

    url_feedback = "https://huggingface.co/datasets/meg-tong/sycophancy-eval/raw/main/feedback.jsonl"
    download_file(url_feedback, "feedback.jsonl")

    # 2. Download Anthropic model-written-evals sycophancy
    url_anthropic = "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_nlp_survey.jsonl"
    download_file(url_anthropic, "anthropic_nlp_survey.jsonl")

    # 3. Copy GSM8K from local candidate assets if available
    local_gsm8k = "C:/Users/ramil/OneDrive/Documents/ChatGPT/repos/audit_candidates/ZeroPromptSearch/Dataset/gsm8k/gsm8k.json"
    dest_gsm8k = os.path.join(DATA_DIR, "gsm8k.json")
    if os.path.exists(local_gsm8k) and not os.path.exists(dest_gsm8k):
        shutil.copyfile(local_gsm8k, dest_gsm8k)
        print(f"[COPIED] GSM8K benchmark dataset ({os.path.getsize(dest_gsm8k)} bytes).")


if __name__ == "__main__":
    main()

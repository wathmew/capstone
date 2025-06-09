import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bertscore
from typing import List, Tuple
from tqdm import tqdm

# Set your non-English language code here (e.g., "zh_Hans" for Chinese, "ar" for Arabic)
SRC_LANG_CODE = "zh_Hans"
TGT_LANG_CODE = "eng_Latn"

# GPU device configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Load NLLB model with device_map auto (supports multiple GPUs automatically)
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(path: str, data: List[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def translate(texts: List[str], batch_size: int = 32) -> List[str]:
    all_translations = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(f"{TGT_LANG_CODE}")
    tokenizer.src_lang = SRC_LANG_CODE

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch,
                           return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move tensors to the same device as the model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend([t.strip() for t in translations])

    return all_translations

def evaluate_pairs(data: List[dict]) -> Tuple[float, float, List[str], List[str]]:
    gold = []
    preds = []

    for entry in tqdm(data, desc="Evaluating"):
        en_trigger, foreign_trigger = entry["trigger_pair"]
        en_args, foreign_args = zip(*entry["argument_pairs"]) if entry["argument_pairs"] else ([], [])

        to_translate = [foreign_trigger] + list(foreign_args)
        translated = translate(to_translate)

        gold_texts = [en_trigger] + list(en_args)
        gold.extend(gold_texts)
        preds.extend(translated)

    # Filter out empty predictions
    filtered = [(g, p) for g, p in zip(gold, preds) if p.strip()]
    if not filtered:
        print("Warning: All predictions were empty.")
        return 0.0, 0.0, gold, preds

    filtered_gold, filtered_preds = zip(*filtered)

    # 1-to-1 exact match accuracy (case insensitive)
    exact_matches = sum(g.strip().lower() == p.strip().lower() for g, p in filtered)
    accuracy = exact_matches / len(filtered)

    # BERTScore
    _, _, bert_f1 = bertscore(filtered_preds, filtered_gold, lang="en", verbose=True, device=DEVICE)
    bert_f1_avg = float(bert_f1.mean())

    return accuracy, bert_f1_avg, gold, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to input JSONL file with trigger_pair and argument_pairs")
    args = parser.parse_args()

    data = load_jsonl(args.input_jsonl)
    accuracy, bert_f1, gold, preds = evaluate_pairs(data)

    # Save translated pairs
    output_path = os.path.splitext(args.input_jsonl)[0] + "_translated.jsonl"
    with open(output_path, "w", encoding="utf-8") as out_f:
        for g, p in zip(gold, preds):
            out_f.write(json.dumps({"gold": g, "pred": p}) + "\n")



    print(f"\nExact Match Accuracy: {accuracy:.4f}")
    print(f"BERTScore (F1): {bert_f1:.4f}")

if __name__ == "__main__":
    main()

import argparse
import json
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='HY_Scoring/fold5')
    parser.add_argument("--data_file", type=str, default='inf_regression.csv')
    parser.add_argument("--item_label_map_file", type=str, default='HY_Scoring/label_mappings.json')
    parser.add_argument("--output_file", type=str, default="inference_results.csv")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def normalize_label(item_label_map, item_id, label):
    scores = sorted(item_label_map[item_id])
    min_val, max_val = min(scores), max(scores)
    return (float(label) - min_val) / (max_val - min_val)


def denormalize_label(item_label_map, item_id, norm_val):
    scores = sorted(item_label_map[item_id])
    min_val, max_val = min(scores), max(scores)
    raw = norm_val * (max_val - min_val) + min_val
    closest = min(scores, key=lambda x: abs(x - raw))
    return closest


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    with open(args.item_label_map_file, 'r', encoding='utf-8') as f:
        item_label_map = json.load(f)

    df = pd.read_csv(args.data_file)
    texts, item_ids, questions, answers = [], [], [], []
    for _, row in df.iterrows():
        if row['item_id'] not in item_label_map:
            continue
        text = f"item_id: {row['item_id']} [SEP] question: {row['question']} [SEP] answer: {row['answer']}"
        texts.append(text)
        item_ids.append(row['item_id'])
        questions.append(row['question'])
        answers.append(row['answer'])

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs).logits.squeeze(-1)
    predictions = outputs.cpu().numpy()

    # Denormalize
    denorm_preds = [denormalize_label(item_label_map, item_id, pred) for item_id, pred in zip(item_ids, predictions)]

    out_df = pd.DataFrame({
        'item_id': item_ids,
        'question': questions,
        'answer': answers,
        'pred_label': denorm_preds
    })
    out_df.to_csv(args.output_file, index=False)
    print(f"Inference completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()

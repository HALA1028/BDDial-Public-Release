import json
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class InferenceEngine:
    def __init__(self, model_path, item_label_map_file, output_file="inference_results.csv", max_length=128, device="cuda:1"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.max_length = max_length
        self.output_file = output_file

        with open(item_label_map_file, 'r', encoding='utf-8') as f:
            self.item_label_map = json.load(f)

    def denormalize_label(self, item_id, norm_val):
        scores = sorted(self.item_label_map[item_id])
        min_val, max_val = min(scores), max(scores)
        raw = norm_val * (max_val - min_val) + min_val
        closest = min(scores, key=lambda x: abs(x - raw))
        return closest

    def infer(self, item_id, question, answer):
        if item_id not in self.item_label_map:
            print(f"[Warning] item_id '{item_id}' not in label map. Skipping.")
            return None

        text = f"item_id: {item_id} [SEP] question: {question} [SEP] answer: {answer}"
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs).logits.squeeze(-1).cpu().item()

        pred_label = self.denormalize_label(item_id, output)

        result = {
            "item_id": item_id,
            "question": question,
            "answer": answer,
            "pred_label": pred_label
        }

        # Append to CSV
        file_exists = os.path.isfile(self.output_file)
        df = pd.DataFrame([result])
        df.to_csv(self.output_file, mode='a', header=not file_exists, index=False)

        print(f"[INFO] Prediction written: {result}")
        return pred_label

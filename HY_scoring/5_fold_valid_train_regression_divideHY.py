# File: fine_tune_macbert_regression_cv.py
import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MacBERT for Regression with 5-Fold Cross Validation")
    parser.add_argument("--primary_data_file", type=str, default='HY_score_without_real_data.csv')
    parser.add_argument("--crossval_dir", type=str, default='private_real_data')
    parser.add_argument("--item_label_map_file", type=str, default='HY_Scoring/label_mappings.json')
    parser.add_argument("--model_name_or_path", type=str, default='hfl/chinese-macbert-large')
    parser.add_argument("--output_dir", type=str, default="./output_regression_Lert_HY_divide")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mae = np.abs(predictions - labels).mean()
    return {"mae": mae}

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_id = args.device.split(":")[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = "cuda:0" #在只能看到一个GPU的情况下，设置为cuda:0
    else:
        args.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    with open(args.item_label_map_file, 'r', encoding='utf-8') as f:
        item_label_map = json.load(f)

    def normalize_label(item_id, label):
        scores = sorted(item_label_map[item_id])
        min_val, max_val = min(scores), max(scores)
        return (float(label) - min_val) / (max_val - min_val)

    def denormalize_label(item_id, norm_val):
        scores = sorted(item_label_map[item_id])
        min_val, max_val = min(scores), max(scores)
        return norm_val * (max_val - min_val) + min_val

    df_primary = pd.read_csv(args.primary_data_file)
    b_files = sorted(glob(os.path.join(args.crossval_dir, "*.csv")))
    assert len(b_files) == 5, "There must be exactly 5 CSV files in the cross-validation directory."

    test_maes = []
    item_mae_summary = defaultdict(list)

    for fold in range(5):
        print(f"===== Fold {fold + 1} =====")

        df_test = pd.read_csv(b_files[fold])
        df_train_parts = [pd.read_csv(f) for i, f in enumerate(b_files) if i != fold]
        df_train_val = pd.concat([df_primary] + df_train_parts, ignore_index=True)
        #df_train_val = pd.concat(df_train_parts, ignore_index=True) #only use the real data for training

        df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=args.seed)

        def tokenize_fn(df):
            texts, labels = [], []
            for _, r in df.iterrows():
                if r['item_id'] not in item_label_map or r['label'] not in item_label_map[r['item_id']]:
                    continue
                text = f"item_id: {r['item_id']} [SEP] question: {r['question']} [SEP] answer: {r['answer']}"
                texts.append(text)
                labels.append(normalize_label(r['item_id'], r['label']))
            tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=args.max_length)
            tokenized['labels'] = labels
            return tokenized

        train_enc = tokenize_fn(df_train)
        val_enc = tokenize_fn(df_val)
        test_enc = tokenize_fn(df_test)

        import datasets
        train_ds = datasets.Dataset.from_dict(train_enc)
        val_ds = datasets.Dataset.from_dict(val_enc)
        test_ds = datasets.Dataset.from_dict(test_enc)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=1
        ).to(args.device)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"fold_{fold + 1}"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(args.output_dir, f"logs_fold_{fold + 1}"),
            load_best_model_at_end=True,
            metric_for_best_model="mae",
            greater_is_better=False,
            seed=args.seed,
            logging_strategy="epoch",
            logging_steps=1
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        trainer.save_model(os.path.join(args.output_dir, f"fold_{fold + 1}"))

        print("Test Evaluation:")
        raw_preds = trainer.predict(test_ds)
        pred_vals = raw_preds.predictions.squeeze()
        true_vals = raw_preds.label_ids

        test_result = compute_metrics((pred_vals, true_vals))
        print(test_result)
        test_maes.append(test_result['mae'])

        # per item_id MAE calculation + output CSV
        predictions = []
        for idx, row in df_test.iterrows():
            if row['item_id'] not in item_label_map or row['label'] not in item_label_map[row['item_id']]:
                continue
            norm_true = normalize_label(row['item_id'], row['label'])
            norm_pred = pred_vals[idx]
            denorm_true = denormalize_label(row['item_id'], norm_true)
            denorm_pred = denormalize_label(row['item_id'], norm_pred)
            error = abs(denorm_pred - denorm_true)
            item_mae_summary[row['item_id']].append(error)
            predictions.append({
                'item_id': row['item_id'],
                'question': row['question'],
                'answer': row['answer'],
                'true_label': denorm_true,
                'pred_label': denorm_pred,
                'abs_error': error
            })

         # add fold_mae to each prediction row
        fold_mae_value = test_result['mae']
        pred_df = pd.DataFrame(predictions)
        pred_df['fold_mae'] = fold_mae_value
        pred_df.to_csv(os.path.join(args.output_dir, f"fold_{fold + 1}_predictions.csv"), index=False)

        # if last fold, write all folds avg mae
        if fold == 4:
            avg_mae = np.mean(test_maes)

            # Calculate denormalized MAE across all folds
            all_denorm_errors = []
            y_errors = []
            h_errors = []

            for idx, row in df_test.iterrows():
                if row['item_id'] not in item_label_map or row['label'] not in item_label_map[row['item_id']]:
                    continue
                norm_true = normalize_label(row['item_id'], row['label'])
                norm_pred = pred_vals[idx]
                denorm_true = denormalize_label(row['item_id'], norm_true)
                denorm_pred = denormalize_label(row['item_id'], norm_pred)
                error = abs(denorm_pred - denorm_true)
                all_denorm_errors.append(error)

                if row['item_id'].startswith('Y'):
                    y_errors.append(error)
                elif row['item_id'].startswith('H'):
                    h_errors.append(error)

            avg_denorm_mae = np.mean(all_denorm_errors)
            avg_y_mae = np.mean(y_errors) if y_errors else float('nan')
            avg_h_mae = np.mean(h_errors) if h_errors else float('nan')

            summary_path = os.path.join(args.output_dir, "all_folds_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Average normalized MAE across all folds: {avg_mae:.4f}\n")
                f.write(f"Average denormalized MAE across all folds: {avg_denorm_mae:.4f}\n")
                f.write(f"Average denormalized MAE for item_ids starting with 'Y': {avg_y_mae:.4f}\n")
                f.write(f"Average denormalized MAE for item_ids starting with 'H': {avg_h_mae:.4f}\n")

    # Save item_id MAE summary
    item_mae_report = {k: float(np.mean(v)) for k, v in item_mae_summary.items()}
    with open(os.path.join(args.output_dir, "item_mae_report.json"), 'w') as f:
        json.dump(item_mae_report, f, indent=2)

if __name__ == "__main__":
    main()

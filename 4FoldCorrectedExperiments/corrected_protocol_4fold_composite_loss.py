#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4-fold corrected protocol with composite CE + focal + dice loss.

This script implements the resubmission experiment variant requested for the
IEEE Access manuscript:

    deduplicate -> 4-fold stratified CV -> train/validation inside each fold
    -> select Youden tau on validation -> evaluate once on the fold test set
    -> aggregate out-of-fold metrics, bootstrap CI, ROC curves, and losses.

It intentionally writes probability scores for every fold so ROC figures can be
regenerated without retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("WANDB_DISABLED", "true")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("datasets") / "multi_disorders"
REDDIT_FILE = "depression_dataset_reddit_cleaned.csv"

SUBSETS = {
    "anger": {
        "file": "AngerIntermittentExplosiveDisorder.csv",
        "pos_label": "Anger/ Intermittent Explosive Disorder",
        "pretty": "Anger/IED (d1)",
    },
    "anxiety": {
        "file": "AnxietyDisorder.csv",
        "pos_label": "Anxiety Disorder",
        "pretty": "Anxiety (d2)",
    },
    "depression": {
        "file": "Depression.csv",
        "pos_label": "Depression",
        "pretty": "Depression (d3)",
    },
    "npd": {
        "file": "NarcissisticDisorder.csv",
        "pos_label": "Narcissistic Disorder",
        "pretty": "NPD (d4)",
    },
    "panic": {
        "file": "PanicDisorder.csv",
        "pos_label": "Panic Disorder",
        "pretty": "Panic (d5)",
    },
}


def normalize(text: str) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[#@]\w+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df["_n"] = df["Text"].map(normalize)
    before = len(df)
    df = df.drop_duplicates(subset="_n", keep="first").reset_index(drop=True)
    return df, before - len(df)


def bootstrap_f1_ci(y_true, y_pred, n_boot: int, seed: int) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(f1_score(y_true[idx], y_pred[idx], average="weighted", zero_division=0))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def youden_threshold(y_true, scores) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_binary_metrics(y_true, scores, tau: float) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_pred = (np.asarray(scores) >= tau).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tau_val": round(float(tau), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(pr), 4),
        "recall": round(float(rc), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


@dataclass
class LossWeights:
    ce: float = 1.0
    focal: float = 1.0
    dice: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


def plot_roc_bundle(name: str, pretty: str, fold_scores: pd.DataFrame, out_dir: Path) -> Path:
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 101)
    interp_tprs = []

    plt.figure(figsize=(5.8, 4.8))
    for fold_id, group in fold_scores.groupby("fold"):
        y = group["y_true"].to_numpy()
        s = group["score_positive"].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1.4, alpha=0.55, label=f"Fold {fold_id} AUC={fold_auc:.3f}")
        interp = np.interp(grid, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)

    y_all = fold_scores["y_true"].to_numpy()
    s_all = fold_scores["score_positive"].to_numpy()
    fpr_all, tpr_all, _ = roc_curve(y_all, s_all)
    auc_all = roc_auc_score(y_all, s_all)
    plt.plot(fpr_all, tpr_all, color="#1d3557", linewidth=2.4, label=f"OOF AUC={auc_all:.3f}")

    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        std_tpr = np.std(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(grid, mean_tpr, color="#2a9d8f", linewidth=2.0, linestyle="-.", label="Mean fold ROC")
        plt.fill_between(
            grid,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color="#2a9d8f",
            alpha=0.15,
            linewidth=0,
        )

    plt.plot([0, 1], [0, 1], color="#8d99ae", linestyle="--", linewidth=1.1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"4-fold ROC: {pretty}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout()
    path = out_dir / f"roc_4fold_{name}_300dpi.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_loss_curves(name: str, pretty: str, log_rows: list[dict], out_dir: Path) -> Path | None:
    frame = pd.DataFrame(log_rows)
    if frame.empty or "loss" not in frame.columns:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 4.3))
    for fold_id, group in frame.dropna(subset=["loss"]).groupby("fold"):
        plt.plot(group["epoch"], group["loss"], marker="o", markersize=3, linewidth=1.4, label=f"Fold {fold_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Composite training loss")
    plt.title(f"Composite loss by fold: {pretty}")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    path = out_dir / f"loss_4fold_{name}_300dpi.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def run_one_disorder(name: str, cfg: dict, args) -> tuple[list[dict], dict, pd.DataFrame, list[dict]]:
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        Trainer,
        TrainingArguments,
    )

    class CompositeLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels_long = labels.long()

            ce_loss = F.cross_entropy(logits, labels_long)
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, labels_long.view(-1, 1)).squeeze(1).clamp(1e-7, 1 - 1e-7)
            focal_loss = (-args.focal_alpha * (1.0 - pt) ** args.focal_gamma * torch.log(pt)).mean()

            pos_probs = probs[:, 1]
            y_pos = labels.float()
            smooth = 1.0
            dice_loss = 1.0 - (2.0 * (pos_probs * y_pos).sum() + smooth) / (
                pos_probs.sum() + y_pos.sum() + smooth
            )

            loss = args.ce_weight * ce_loss + args.focal_weight * focal_loss + args.dice_weight * dice_loss
            return (loss, outputs) if return_outputs else loss

    path = DATA_DIR / cfg["file"]
    df = pd.read_csv(path)
    df, removed = deduplicate(df)
    df["y"] = (df["label"] == cfg["pos_label"]).astype(int)
    df["row_id"] = np.arange(len(df))

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

    def make_ds(frame: pd.DataFrame):
        ds = Dataset.from_pandas(
            frame[["Text", "y"]].rename(columns={"y": "label"}),
            preserve_index=False,
        )
        ds = ds.map(
            lambda e: tokenizer(e["Text"], truncation=True, padding="max_length", max_length=args.max_length),
            batched=True,
        )
        ds = ds.remove_columns(["Text"])
        ds.set_format("torch")
        return ds

    def trainer_metrics(p):
        preds = p.predictions.argmax(-1)
        return compute_binary_metrics(p.label_ids, p.predictions[:, 1], 0.5) | {"pred_f1": 0.0}

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows: list[dict] = []
    all_score_frames: list[pd.DataFrame] = []
    loss_rows: list[dict] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(cv.split(df["Text"], df["y"]), start=1):
        train_val = df.iloc[train_val_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        train, val = train_test_split(
            train_val,
            test_size=args.inner_val_size,
            random_state=args.seed + fold_idx,
            stratify=train_val["y"],
        )
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

        print(f"\n>>> {cfg['pretty']} fold {fold_idx}/{args.folds}")
        print(f"    dedup_removed={removed} n_train={len(train)} n_val={len(val)} n_test={len(test)}")

        ds_train = make_ds(train)
        ds_val = make_ds(val)
        ds_test = make_ds(test)

        config = DistilBertConfig.from_pretrained(
            args.model_name,
            num_labels=2,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
        model = DistilBertForSequenceClassification.from_pretrained(args.model_name, config=config)

        targs = TrainingArguments(
            output_dir=f"./trained_models_4fold/{name}/fold_{fold_idx}",
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            load_best_model_at_end=False,
            report_to=[],
            seed=args.seed + fold_idx,
        )
        trainer = CompositeLossTrainer(
            model=model,
            args=targs,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=tokenizer,
            compute_metrics=trainer_metrics,
        )
        trainer.train()

        for item in trainer.state.log_history:
            if "loss" in item or "eval_loss" in item:
                row = dict(item)
                row["disorder"] = cfg["pretty"]
                row["fold"] = fold_idx
                loss_rows.append(row)

        def positive_scores(ds):
            logits = trainer.predict(ds).predictions
            probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            return probs

        val_scores = positive_scores(ds_val)
        test_scores = positive_scores(ds_test)
        tau = youden_threshold(val["y"].to_numpy(), val_scores)
        metrics = compute_binary_metrics(test["y"].to_numpy(), test_scores, tau)

        fold_frame = pd.DataFrame({
            "disorder_key": name,
            "disorder": cfg["pretty"],
            "fold": fold_idx,
            "row_id": test["row_id"].to_numpy(),
            "y_true": test["y"].to_numpy(),
            "score_positive": test_scores,
            "tau_val": tau,
            "y_pred_at_tau_val": (test_scores >= tau).astype(int),
        })
        all_score_frames.append(fold_frame)

        fold_row = {
            "disorder_key": name,
            "disorder": cfg["pretty"],
            "fold": fold_idx,
            "dedup_removed": removed,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            **metrics,
        }
        fold_rows.append(fold_row)
        print(
            f"    tau={tau:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} TP={metrics['TP']} FP={metrics['FP']} "
            f"FN={metrics['FN']} TN={metrics['TN']}"
        )

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_frame = pd.concat(all_score_frames, ignore_index=True)
    y_all = score_frame["y_true"].to_numpy()
    s_all = score_frame["score_positive"].to_numpy()
    tau_per_row = score_frame["tau_val"].to_numpy()
    pred_all = (s_all >= tau_per_row).astype(int)
    oof_metrics = compute_binary_metrics(y_all, s_all, 0.5)
    oof_metrics.pop("tau_val", None)
    # Fold-specific tau values are applied for the reported OOF classification metrics.
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    pr, rc, f1, _ = precision_recall_fscore_support(y_all, pred_all, average="weighted", zero_division=0)
    cm = confusion_matrix(y_all, pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1_lo, f1_hi = bootstrap_f1_ci(y_all, pred_all, args.bootstrap, args.seed)
    summary = {
        "disorder_key": name,
        "disorder": cfg["pretty"],
        "folds": args.folds,
        "dedup_removed": removed,
        "n_total_dedup": len(df),
        "f1_fold_mean": round(float(pd.DataFrame(fold_rows)["f1"].mean()), 4),
        "f1_fold_std": round(float(pd.DataFrame(fold_rows)["f1"].std(ddof=1)), 4),
        "auc_fold_mean": round(float(pd.DataFrame(fold_rows)["auc"].mean()), 4),
        "auc_fold_std": round(float(pd.DataFrame(fold_rows)["auc"].std(ddof=1)), 4),
        "oof_f1": round(float(f1), 4),
        "oof_f1_ci_lo": f1_lo,
        "oof_f1_ci_hi": f1_hi,
        "oof_auc": round(float(roc_auc_score(y_all, s_all)), 4),
        "oof_accuracy": round(float(accuracy_score(y_all, pred_all)), 4),
        "oof_precision": round(float(pr), 4),
        "oof_recall": round(float(rc), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "tau_fold_values": ";".join(f"{v:.4f}" for v in pd.DataFrame(fold_rows)["tau_val"].to_list()),
    }
    return fold_rows, summary, score_frame, loss_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disorders", nargs="*", default=list(SUBSETS.keys()), choices=list(SUBSETS.keys()))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--inner-val-size", type=float, default=0.20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--warmup-steps", type=int, default=24)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--focal-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--out-dir", default="fourfold_composite_outputs")
    args = ap.parse_args()

    if not DATA_DIR.is_dir():
        sys.exit(f"ERROR: {DATA_DIR} not found. Run from the repository root.")

    out_dir = Path(args.out_dir)
    roc_dir = out_dir / "roc_300dpi"
    loss_dir = out_dir / "loss_300dpi"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("4-FOLD CORRECTED PROTOCOL WITH COMPOSITE LOSS")
    print("=" * 72)
    print(json.dumps(vars(args), indent=2))

    all_fold_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    all_scores: list[pd.DataFrame] = []
    all_loss_rows: list[dict] = []
    figure_rows: list[dict] = []

    for name in args.disorders:
        cfg = SUBSETS[name]
        fold_rows, summary, score_frame, loss_rows = run_one_disorder(name, cfg, args)
        all_fold_rows.extend(fold_rows)
        all_summary_rows.append(summary)
        all_scores.append(score_frame)
        all_loss_rows.extend(loss_rows)

        score_path = out_dir / f"{name}_4fold_oof_scores.csv"
        score_frame.to_csv(score_path, index=False)
        roc_path = plot_roc_bundle(name, cfg["pretty"], score_frame, roc_dir)
        loss_path = plot_loss_curves(name, cfg["pretty"], loss_rows, loss_dir)
        figure_rows.append({
            "disorder": cfg["pretty"],
            "roc_figure": str(roc_path),
            "loss_figure": str(loss_path) if loss_path else "",
            "scores_csv": str(score_path),
        })

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    scores_df = pd.concat(all_scores, ignore_index=True)
    loss_df = pd.DataFrame(all_loss_rows)
    figures_df = pd.DataFrame(figure_rows)

    fold_df.to_csv(out_dir / "fourfold_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "fourfold_summary_metrics.csv", index=False)
    scores_df.to_csv(out_dir / "fourfold_all_oof_scores.csv", index=False)
    loss_df.to_csv(out_dir / "fourfold_loss_history.csv", index=False)
    figures_df.to_csv(out_dir / "fourfold_artifact_manifest.csv", index=False)

    print("\n" + "=" * 72)
    print("4-fold summary")
    print(summary_df.to_string(index=False))
    print("=" * 72)
    print(f"Outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

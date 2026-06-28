#!/usr/bin/env python3
"""Build final notebook, extra figures, and manifest for 4-fold artifacts."""

from __future__ import annotations

import json
import platform
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "fourfold_composite_outputs"
FIG = OUT / "summary_300dpi"
NOTEBOOK_DIR = OUT / "notebooks"
LOG = ROOT / "run_4fold_composite_gpu.log"
ERR = ROOT / "run_4fold_composite_gpu.err.log"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_summary(summary: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    labels = summary["disorder"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(8.4, 5.0))
    y = summary["oof_f1"].to_numpy()
    lo = y - summary["oof_f1_ci_lo"].to_numpy()
    hi = summary["oof_f1_ci_hi"].to_numpy() - y
    plt.errorbar(x, y, yerr=[lo, hi], fmt="o", capsize=5, color="#1d3557")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Out-of-fold weighted F1")
    plt.ylim(max(0, min(summary["oof_f1_ci_lo"]) - 0.08), 1.02)
    plt.title("4-fold corrected protocol: OOF F1 with bootstrap 95% CI")
    for i, value in enumerate(y):
        plt.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    p = FIG / "fourfold_oof_f1_bootstrap_ci_300dpi.png"
    savefig(p)
    paths.append(p)

    width = 0.35
    plt.figure(figsize=(8.4, 5.0))
    plt.bar(x - width / 2, summary["f1_fold_mean"], width, label="Mean fold F1", color="#2a9d8f")
    plt.bar(x + width / 2, summary["oof_f1"], width, label="OOF F1", color="#1d3557")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Weighted F1")
    plt.ylim(0, 1.05)
    plt.legend(frameon=False)
    plt.title("Fold-mean F1 vs pooled out-of-fold F1")
    p = FIG / "fourfold_foldmean_vs_oof_f1_300dpi.png"
    savefig(p)
    paths.append(p)

    plt.figure(figsize=(8.4, 5.0))
    plt.bar(x, summary["oof_auc"], color="#6d597a")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Out-of-fold ROC AUC")
    plt.ylim(0, 1.05)
    plt.title("4-fold corrected protocol: OOF ROC AUC")
    for i, value in enumerate(summary["oof_auc"]):
        plt.text(i, value + 0.018, f"{value:.3f}", ha="center", fontsize=9)
    p = FIG / "fourfold_oof_auc_300dpi.png"
    savefig(p)
    paths.append(p)

    for _, row in summary.iterrows():
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        plt.figure(figsize=(4.2, 3.8))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111", fontsize=13)
        safe = row["disorder_key"]
        plt.title(f"4-fold OOF confusion: {row['disorder']}")
        p = FIG / f"confusion_4fold_oof_{safe}_300dpi.png"
        savefig(p)
        paths.append(p)
    return paths


def table_md(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_markdown(index=False)


def tail(path: Path, n: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def create_notebook(summary: pd.DataFrame, fold: pd.DataFrame, figure_paths: list[Path]) -> Path:
    env = {"platform": platform.platform()}
    try:
        import torch
        import transformers

        env.update({
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        env["env_error"] = repr(exc)

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 4-Fold Corrected Experiments with Composite Loss\n\n"
            "This notebook documents the revised experiment run: exact deduplication, "
            "4-fold stratified cross-validation, a validation split inside each fold for Youden tau, "
            "and the composite training objective `CE + focal + dice`."
        ),
        nbf.v4.new_markdown_cell(f"## Runtime Environment\n\n```json\n{json.dumps(env, indent=2)}\n```"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd() / 'fourfold_composite_outputs'\n"
            "summary = pd.read_csv(ROOT / 'fourfold_summary_metrics.csv')\n"
            "fold = pd.read_csv(ROOT / 'fourfold_fold_metrics.csv')\n"
            "summary, fold.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## Summary Metrics\n\n"
            + table_md(
                summary,
                [
                    "disorder",
                    "dedup_removed",
                    "n_total_dedup",
                    "f1_fold_mean",
                    "f1_fold_std",
                    "oof_f1",
                    "oof_f1_ci_lo",
                    "oof_f1_ci_hi",
                    "oof_auc",
                    "oof_accuracy",
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                ],
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Fold-Level Metrics\n\n"
            + table_md(
                fold,
                [
                    "disorder",
                    "fold",
                    "n_train",
                    "n_val",
                    "n_test",
                    "tau_val",
                    "f1",
                    "auc",
                    "accuracy",
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                ],
            )
        ),
        nbf.v4.new_markdown_cell(
            "## 300 dpi Figures\n\n"
            + "\n\n".join(
                f"![{p.stem}]({p.relative_to(OUT).as_posix()})"
                for p in figure_paths
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Re-run Command\n\n"
            "```powershell\n"
            "py -3.10 corrected_protocol_4fold_composite_loss.py --epochs 4 --batch-size 4 --bootstrap 2000 --out-dir fourfold_composite_outputs\n"
            "```"
        ),
        nbf.v4.new_markdown_cell("## GPU Run Log Tail\n\n```text\n" + tail(LOG) + "\n```"),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / "Ovcharuk_IEEE_Access_4fold_composite_loss_executed.ipynb"
    nbf.write(nb, path)
    return path


def zip_outputs(zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [ROOT / "corrected_protocol_4fold_composite_loss.py", ROOT / "build_4fold_submission_artifacts.py", LOG, ERR]:
            if path.exists():
                zf.write(path, path.name)
        for path in OUT.rglob("*"):
            if path.is_file():
                zf.write(path, Path("fourfold_composite_outputs") / path.relative_to(OUT))


def main() -> None:
    summary = pd.read_csv(OUT / "fourfold_summary_metrics.csv")
    fold = pd.read_csv(OUT / "fourfold_fold_metrics.csv")
    figure_paths = []
    figure_paths.extend(sorted((OUT / "roc_300dpi").glob("*.png")))
    figure_paths.extend(sorted((OUT / "loss_300dpi").glob("*.png")))
    figure_paths.extend(plot_summary(summary))
    notebook = create_notebook(summary, fold, figure_paths)

    manifest = {
        "summary_csv": str(OUT / "fourfold_summary_metrics.csv"),
        "fold_metrics_csv": str(OUT / "fourfold_fold_metrics.csv"),
        "all_oof_scores_csv": str(OUT / "fourfold_all_oof_scores.csv"),
        "loss_history_csv": str(OUT / "fourfold_loss_history.csv"),
        "notebook": str(notebook),
        "figures_300dpi": [str(p) for p in figure_paths],
        "run_log": str(LOG),
        "run_err": str(ERR),
        "loss": "CE + focal(alpha=0.25,gamma=2.0) + dice, weights 1/1/1",
        "cv": "StratifiedKFold(n_splits=4), inner validation split 0.20 from train_val for tau selection",
    }
    manifest_path = OUT / "fourfold_final_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = ROOT / "Ovcharuk_4fold_composite_loss_GPU_artifacts.zip"
    zip_outputs(zip_path)
    print(json.dumps(manifest | {"zip": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

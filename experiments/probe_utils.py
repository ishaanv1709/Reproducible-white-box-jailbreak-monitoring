"""
Shared utilities for probing experiments:
  - Hidden-state extraction from any HuggingFace causal LM
  - Logistic regression probe with calibration
  - Standard probing metrics (AUROC, ECE, F1, etc.)
  - Bootstrap confidence intervals
  - Plotting helpers (uniform style across both experiments)
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42

# ---------------------------------------------------------------------------
# Plot styling — applied once on import.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

PALETTE = {
    "primary": "#1f77b4",
    "accent": "#d62728",
    "neutral": "#7f7f7f",
    "good": "#2ca02c",
    "warm": "#ff7f0e",
    "violet": "#9467bd",
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Hidden-state extraction (last non-padding token, all layers)
# ---------------------------------------------------------------------------

def extract_hidden_states(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 4,
    max_length: int = 256,
    desc: str = "Extracting hidden states",
) -> np.ndarray:
    """Extract last-non-padding-token hidden states across all layers.

    Returns array of shape (N, num_layers, hidden_dim). Includes the embedding
    output as layer 0; transformer layers follow.
    """
    model.eval()
    out_chunks: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )

        # outputs.hidden_states: tuple of (B, T, H), one per layer (incl. embeds)
        attn_mask = enc["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1  # (B,) — last non-padding position
        bsz = enc["input_ids"].shape[0]
        batch_idx = torch.arange(bsz, device=device)

        per_layer = []
        for h in outputs.hidden_states:
            # h: (B, T, H) — pick last non-pad token
            last = h[batch_idx, last_idx]  # (B, H)
            per_layer.append(last.detach().to(torch.float32).cpu().numpy())

        # Stack to (B, num_layers, H)
        out_chunks.append(np.stack(per_layer, axis=1))

    return np.concatenate(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    seed: int = SEED,
):
    """Train a regularized, class-balanced logistic regression probe."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(
        max_iter=4000,
        C=C,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
    )
    probe.fit(X_scaled, y)
    return probe, scaler


def probe_predict_proba(probe, scaler, X: np.ndarray) -> np.ndarray:
    return probe.predict_proba(scaler.transform(X))[:, 1]


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-bin ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        if i == 0:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = float(y[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def evaluate_probe(probe, scaler, X: np.ndarray, y: np.ndarray) -> dict:
    probs = probe_predict_proba(probe, scaler, X)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "brier": float(brier_score_loss(y, probs)),
        "ece": expected_calibration_error(probs, y),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }


def bootstrap_auroc(
    probs: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = SEED
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for 95% bootstrap CI on AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y[idx], probs[idx]))
    if not aurocs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(aurocs)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_layer_auroc(
    layer_aurocs: Sequence[float],
    multi_auroc: float,
    out_path: str,
    title: str = "Layer-wise probe performance",
    multilayer_label: str = "Multi-layer aggregated",
    extra_curves: dict[str, Sequence[float]] | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    layers = np.arange(len(layer_aurocs))
    ax.plot(
        layers, layer_aurocs,
        color=PALETTE["primary"], marker="o",
        label="Single-layer probe", zorder=3,
    )
    if extra_curves:
        colors = [PALETTE["warm"], PALETTE["violet"], PALETTE["good"]]
        for (name, curve), c in zip(extra_curves.items(), colors):
            ax.plot(layers, curve, color=c, marker="s", markersize=4, label=name, zorder=2)
    ax.axhline(multi_auroc, color=PALETTE["accent"], linestyle="--",
               label=f"{multilayer_label}: {multi_auroc:.3f}", zorder=4)
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle=":",
               label="Random baseline (0.50)", zorder=1)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(layers[::max(1, len(layers) // 12)])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(
    probs: np.ndarray, y: np.ndarray, out_path: str, title: str = "Reliability diagram"
):
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins)
    for i in range(n_bins):
        if i == 0:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs > edges[i]) & (probs <= edges[i + 1])
        bin_count[i] = mask.sum()
        if bin_count[i] > 0:
            bin_acc[i] = y[mask].mean()
            bin_conf[i] = probs[mask].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # Reliability
    ax1.plot([0, 1], [0, 1], color=PALETTE["neutral"], linestyle=":", label="Perfect calibration")
    valid = ~np.isnan(bin_acc)
    ax1.plot(centers[valid], bin_acc[valid], "o-", color=PALETTE["primary"], label="Probe")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical positive rate")
    ax1.set_title("Calibration (reliability diagram)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")

    # Confidence histogram
    ax2.bar(centers, bin_count, width=(edges[1] - edges[0]) * 0.92,
            color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")
    ax2.set_xlim(0, 1)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(preds: np.ndarray, y: np.ndarray, out_path: str,
                   labels: tuple[str, str] = ("Negative", "Positive"),
                   title: str = "Confusion matrix"):
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca_embedding(
    hidden_layer: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    label_names: dict[int, str] | None = None,
    title: str = "PCA of hidden-state representations",
):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED)
    pts = pca.fit_transform(hidden_layer)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    label_names = label_names or {int(v): str(v) for v in np.unique(labels)}
    palette = [PALETTE["primary"], PALETTE["accent"], PALETTE["good"],
               PALETTE["warm"], PALETTE["violet"]]
    for i, v in enumerate(sorted(np.unique(labels))):
        m = labels == v
        ax.scatter(
            pts[m, 0], pts[m, 1],
            s=22, alpha=0.78, color=palette[i % len(palette)],
            label=label_names.get(int(v), str(v)),
            edgecolor="white", linewidth=0.4,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_metrics_table(
    rows: list[dict], columns: list[str], out_path: str
):
    """Write a CSV table with the given columns."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c, "")) for c in columns) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)

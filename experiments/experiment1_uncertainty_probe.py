"""
Experiment 1 — Uncertainty Probing on Monitor Model Hidden States
=================================================================

Question. Suppose a small monitor model classifies a frontier model's
outputs as safe or unsafe. Do its hidden states encode a reliable
"I'm not sure" signal that is recoverable with a linear probe?

Method.
1. We build a 3-bucket dataset: clearly-safe, clearly-unsafe, and genuinely
   ambiguous queries (dual-use cybersecurity, contextual edge cases).
2. We treat clearly-safe ∪ clearly-unsafe as 'certain' (label 0) and
   ambiguous as 'uncertain' (label 1) — the operational signal a monitor
   would need to abstain or escalate.
3. For a small open monitor model (Qwen/Qwen2.5-1.5B-Instruct), we extract
   last-token hidden states across every transformer layer.
4. We train a regularised, class-balanced logistic-regression probe
   per layer, plus a multi-layer aggregated probe.
5. We evaluate AUROC, ECE, F1, calibration, layer-wise depth, and a PCA
   projection of the geometric structure.

The aim: a clean, fully reproducible, bug-tolerant pilot that mirrors the
white-box-monitoring methodology described in Direction 2.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Local imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from datasets import build_experiment1_dataset  # noqa: E402
from probe_utils import (  # noqa: E402
    SEED,
    bootstrap_auroc,
    evaluate_probe,
    extract_hidden_states,
    fit_probe,
    plot_calibration,
    plot_confusion,
    plot_layer_auroc,
    plot_pca_embedding,
    probe_predict_proba,
    save_metrics_table,
    set_seed,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = HERE / "results" / "experiment1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 4 if DEVICE == "cuda" else 2
MAX_LENGTH = 256


def main() -> None:
    set_seed(SEED)
    print(f"[setup] device={DEVICE}  dtype={DTYPE}  model={MODEL_NAME}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    texts, uncertain_labels, safety_labels = build_experiment1_dataset()
    uncertain_labels = np.array(uncertain_labels)
    safety_labels = np.array(safety_labels)

    n_safe = int((safety_labels == 0).sum())
    n_unsafe = int((safety_labels == 1).sum())
    n_ambig = int((safety_labels == -1).sum())
    print(f"[data] N={len(texts)}  safe={n_safe}  unsafe={n_unsafe}  ambiguous={n_ambig}")

    # ------------------------------------------------------------------
    # Load monitor model
    # ------------------------------------------------------------------
    print(f"[model] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------
    H = extract_hidden_states(
        texts, tokenizer, model,
        device=DEVICE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        desc="Hidden states (Exp 1)",
    )
    print(f"[states] shape={H.shape}  (N, num_layers, hidden_dim)")
    n_layers = H.shape[1]

    # Free GPU memory before probing
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Train/test split — stratified on uncertainty label
    # ------------------------------------------------------------------
    indices = np.arange(len(texts))
    (H_train, H_test, y_train, y_test,
     safety_train, safety_test, idx_train, idx_test) = train_test_split(
        H, uncertain_labels, safety_labels, indices,
        test_size=0.30, stratify=uncertain_labels, random_state=SEED,
    )
    train_texts = [texts[i] for i in idx_train]
    test_texts = [texts[i] for i in idx_test]
    print(f"[split] train={len(y_train)}  test={len(y_test)}  "
          f"test+={int(y_test.sum())}  test-={int((1 - y_test).sum())}")

    # ------------------------------------------------------------------
    # Per-layer probes
    # ------------------------------------------------------------------
    layer_metrics: list[dict] = []
    layer_aurocs: list[float] = []
    for L in range(n_layers):
        probe, scaler = fit_probe(H_train[:, L, :], y_train)
        m = evaluate_probe(probe, scaler, H_test[:, L, :], y_test)
        m["layer"] = L
        layer_metrics.append(m)
        layer_aurocs.append(m["auroc"])
        if L % max(1, n_layers // 8) == 0 or L == n_layers - 1:
            print(f"[probe] layer {L:2d}: AUROC={m['auroc']:.3f}  "
                  f"F1={m['f1']:.3f}  ECE={m['ece']:.3f}")

    best_layer = int(np.argmax(layer_aurocs))
    print(f"[probe] best single-layer: layer {best_layer}, "
          f"AUROC={layer_aurocs[best_layer]:.3f}")

    # ------------------------------------------------------------------
    # Multi-layer aggregated probe (concat all layer features)
    # ------------------------------------------------------------------
    X_tr_all = H_train.reshape(H_train.shape[0], -1)
    X_te_all = H_test.reshape(H_test.shape[0], -1)
    multi_probe, multi_scaler = fit_probe(X_tr_all, y_train, C=0.5)
    multi_metrics = evaluate_probe(multi_probe, multi_scaler, X_te_all, y_test)
    multi_probs = probe_predict_proba(multi_probe, multi_scaler, X_te_all)
    multi_preds = (multi_probs >= 0.5).astype(int)
    print(f"[multi] AUROC={multi_metrics['auroc']:.3f}  "
          f"F1={multi_metrics['f1']:.3f}  ECE={multi_metrics['ece']:.3f}")

    # Bootstrap CI on multi-layer AUROC
    auroc_mean, auroc_lo, auroc_hi = bootstrap_auroc(multi_probs, y_test)
    print(f"[multi] AUROC bootstrap 95% CI: [{auroc_lo:.3f}, {auroc_hi:.3f}]")

    # ------------------------------------------------------------------
    # Top-K layer aggregation curve (use top 1, top 3, top 5, top 8 layers)
    # ------------------------------------------------------------------
    layer_order = list(np.argsort(layer_aurocs)[::-1])
    topk_aurocs: dict[str, float] = {}
    for k in [1, 3, 5, 8]:
        if k > n_layers:
            continue
        chosen = layer_order[:k]
        X_tr_k = H_train[:, chosen, :].reshape(H_train.shape[0], -1)
        X_te_k = H_test[:, chosen, :].reshape(H_test.shape[0], -1)
        p_k, s_k = fit_probe(X_tr_k, y_train, C=0.5)
        m_k = evaluate_probe(p_k, s_k, X_te_k, y_test)
        topk_aurocs[f"top-{k}"] = m_k["auroc"]
        print(f"[topk] top-{k} layers (idx={chosen}): AUROC={m_k['auroc']:.3f}")

    # ------------------------------------------------------------------
    # TF-IDF lexical baseline — guards against the "probe just learns topic"
    # critique. If TF-IDF on raw text matches the probe, the linear-probe
    # finding may be partly explained by surface lexical cues. The probe is
    # only doing something deeper than topic-detection if it clearly beats
    # this baseline.
    # ------------------------------------------------------------------
    print("[tfidf] training TF-IDF + LR baseline on raw text")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            sublinear_tf=True, lowercase=True)
    X_tr_tfidf = tfidf.fit_transform(train_texts)
    X_te_tfidf = tfidf.transform(test_texts)
    tfidf_clf = LogisticRegression(
        max_iter=4000, C=1.0, class_weight="balanced",
        solver="liblinear", random_state=SEED,
    )
    tfidf_clf.fit(X_tr_tfidf, y_train)
    tfidf_probs = tfidf_clf.predict_proba(X_te_tfidf)[:, 1]
    tfidf_preds = (tfidf_probs >= 0.5).astype(int)
    from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                                 brier_score_loss)
    from probe_utils import expected_calibration_error
    tfidf_metrics = {
        "auroc": float(roc_auc_score(y_test, tfidf_probs)),
        "accuracy": float(accuracy_score(y_test, tfidf_preds)),
        "f1": float(f1_score(y_test, tfidf_preds, zero_division=0)),
        "brier": float(brier_score_loss(y_test, tfidf_probs)),
        "ece": expected_calibration_error(tfidf_probs, y_test),
    }
    print(f"[tfidf] AUROC={tfidf_metrics['auroc']:.3f}  "
          f"F1={tfidf_metrics['f1']:.3f}  ECE={tfidf_metrics['ece']:.3f}")
    probe_minus_tfidf = multi_metrics["auroc"] - tfidf_metrics["auroc"]
    print(f"[tfidf] probe AUROC - TF-IDF AUROC = {probe_minus_tfidf:+.3f}")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    plot_layer_auroc(
        layer_aurocs, multi_metrics["auroc"],
        str(RESULTS_DIR / "fig_layer_auroc.png"),
        title="Experiment 1 — Layer-wise uncertainty probe AUROC",
    )

    # Probe vs TF-IDF baseline bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = [
        ("Layer 0 (embedding)\nbaseline", layer_metrics[0]["auroc"], "#7f7f7f"),
        ("TF-IDF + LR\n(lexical baseline)", tfidf_metrics["auroc"], "#ff7f0e"),
        (f"Best single layer\n(layer {best_layer})", layer_aurocs[best_layer], "#1f77b4"),
        ("Multi-layer\naggregated probe", multi_metrics["auroc"], "#2ca02c"),
    ]
    xs = np.arange(len(methods))
    bars = ax.bar(xs, [m[1] for m in methods],
                  color=[m[2] for m in methods],
                  edgecolor="white", linewidth=1.4)
    for i, (_, v, _) in enumerate(methods):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", label="Random (0.50)")
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.08)
    ax.set_title("Experiment 1 — Uncertainty detection: probe vs lexical baseline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "fig_baseline_comparison.png"))
    plt.close(fig)

    plot_calibration(
        multi_probs, y_test,
        str(RESULTS_DIR / "fig_calibration.png"),
        title="Experiment 1 — Multi-layer probe calibration",
    )

    plot_confusion(
        multi_preds, y_test,
        str(RESULTS_DIR / "fig_confusion.png"),
        labels=("Certain", "Uncertain"),
        title="Experiment 1 — Multi-layer probe confusion matrix",
    )

    # PCA at the best single layer, coloured by 3-class safety/uncertainty
    plot_pca_embedding(
        H[:, best_layer, :],
        # Map: 0 safe, 1 unsafe, 2 ambiguous
        np.array([
            0 if s == 0 else (1 if s == 1 else 2)
            for s in safety_labels
        ]),
        str(RESULTS_DIR / "fig_pca.png"),
        label_names={0: "Clear safe", 1: "Clear unsafe", 2: "Ambiguous"},
        title=f"Experiment 1 — PCA of hidden states (layer {best_layer})",
    )

    # ------------------------------------------------------------------
    # Save metrics & layer table
    # ------------------------------------------------------------------
    table_columns = ["layer", "auroc", "accuracy", "precision", "recall", "f1", "brier", "ece"]
    save_metrics_table(layer_metrics, table_columns,
                       str(RESULTS_DIR / "table_layer_metrics.csv"))

    summary = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "n_samples": int(len(texts)),
        "split": {"train": int(len(y_train)), "test": int(len(y_test))},
        "buckets": {"safe": n_safe, "unsafe": n_unsafe, "ambiguous": n_ambig},
        "n_layers": int(n_layers),
        "best_single_layer": {
            "index": best_layer,
            "auroc": float(layer_aurocs[best_layer]),
            "metrics": layer_metrics[best_layer],
        },
        "multilayer": {
            "metrics": multi_metrics,
            "auroc_bootstrap_95ci": [auroc_lo, auroc_hi],
        },
        "topk_aurocs": topk_aurocs,
        "tfidf_baseline": tfidf_metrics,
        "probe_vs_tfidf_auroc_gap": float(probe_minus_tfidf),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] artifacts written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

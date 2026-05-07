"""
Experiment 2 — Linear-Probe Detection of Adversarial Context Injection
======================================================================

Question. An adversary injects crafted content into the input context of an
AI system, attempting to make it produce recommendations that contradict
its trusted source. With white-box access to a smaller gateway model, can a
linear probe on its hidden states detect these adversarial inputs?

Method.
1. Paired-context dataset of 50 trusted-source scenarios. Each scenario
   yields four variants (200 samples total):
       - clean: trusted source + benign user task
       - direct override: explicit "ignore previous" instruction
       - indirect contradiction: a plausible sentence that contradicts the source
       - obfuscated: register-shifted framing of the same override
2. Two architecturally distinct gateway models, each <4B parameters:
       - HuggingFaceTB/SmolLM2-1.7B-Instruct  (SmolLM family, 1.7B)
       - microsoft/Phi-3.5-mini-instruct       (Phi family, 3.8B)
3. For each model, extract last-non-padding-token hidden states across all
   transformer layers.
4. Train regularised, class-balanced logistic-regression probes:
       - Per-layer probes
       - Multi-layer aggregated probe
       - Cross-strategy generalisation (train on N-1 strategies, test on held-out)
       - Cross-model probe transfer via PCA-shared subspace
5. Evaluate AUROC, F1, ECE, layer-wise depth, plus PCA visualisation of the
   geometric structure of clean vs injected representations.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from datasets import build_experiment2_dataset  # noqa: E402
from probe_utils import (  # noqa: E402
    PALETTE,
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

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS = {
    "smollm2":   "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen15b":   "Qwen/Qwen2.5-1.5B-Instruct",
}

RESULTS_DIR = HERE / "results" / "experiment2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 4 if DEVICE == "cuda" else 2
MAX_LENGTH = 320


def _load_model(model_id: str):
    print(f"[model] loading {model_id}")
    needs_remote = "phi-3" in model_id.lower() or "phi3" in model_id.lower()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=needs_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=needs_remote,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    mdl.eval()
    return tok, mdl


def _free(model) -> None:
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def run_per_model(
    model_key: str,
    model_id: str,
    texts: Sequence[str],
    labels: np.ndarray,
    strategies: np.ndarray,
) -> dict:
    """Extract states from one model, run the full probing suite, return summary."""
    out: dict = {"model": model_id, "key": model_key}
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- hidden states -------------------------------------------------
    tok, mdl = _load_model(model_id)
    H = extract_hidden_states(
        texts, tok, mdl,
        device=DEVICE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        desc=f"Hidden states ({model_key})",
    )
    _free(mdl)
    print(f"[{model_key}] H shape={H.shape}")
    n_layers = H.shape[1]
    out["hidden_shape"] = list(H.shape)

    # --- split (stratified on strategy to balance attack types) --------
    H_tr, H_te, y_tr, y_te, strat_tr, strat_te = train_test_split(
        H, labels, strategies,
        test_size=0.30, stratify=labels, random_state=SEED,
    )
    out["split"] = {"train": int(len(y_tr)), "test": int(len(y_te))}

    # --- per-layer probes ---------------------------------------------
    layer_metrics: list[dict] = []
    layer_aurocs: list[float] = []
    for L in range(n_layers):
        p, s = fit_probe(H_tr[:, L, :], y_tr)
        m = evaluate_probe(p, s, H_te[:, L, :], y_te)
        m["layer"] = L
        layer_metrics.append(m)
        layer_aurocs.append(m["auroc"])
    best_layer = int(np.argmax(layer_aurocs))
    print(f"[{model_key}] best single layer: {best_layer}  AUROC={layer_aurocs[best_layer]:.3f}")
    out["layer_aurocs"] = layer_aurocs
    out["best_single_layer"] = {"index": best_layer, "auroc": float(layer_aurocs[best_layer])}

    # --- multi-layer aggregated probe ---------------------------------
    X_tr_all = H_tr.reshape(H_tr.shape[0], -1)
    X_te_all = H_te.reshape(H_te.shape[0], -1)
    p_all, s_all = fit_probe(X_tr_all, y_tr, C=0.5)
    multi_metrics = evaluate_probe(p_all, s_all, X_te_all, y_te)
    multi_probs = probe_predict_proba(p_all, s_all, X_te_all)
    multi_preds = (multi_probs >= 0.5).astype(int)
    print(f"[{model_key}] multi-layer: AUROC={multi_metrics['auroc']:.3f}")
    out["multilayer"] = multi_metrics

    # --- cross-strategy generalisation --------------------------------
    # For each held-out injection strategy, train probe on (clean ∪ remaining
    # injection strategies) and evaluate on (clean ∪ held-out strategy) at the
    # best single layer.
    cross_results: dict[str, dict] = {}
    inject_strategies = ["direct", "indirect", "obfuscated"]
    for held_out in inject_strategies:
        train_mask = np.array([
            (s == "clean") or (s in inject_strategies and s != held_out)
            for s in strat_tr
        ])
        test_mask = np.array([
            (s == "clean") or (s == held_out)
            for s in strat_te
        ])
        if train_mask.sum() < 5 or test_mask.sum() < 5:
            continue
        X_tr_g = H_tr[train_mask, best_layer, :]
        y_tr_g = y_tr[train_mask]
        X_te_g = H_te[test_mask, best_layer, :]
        y_te_g = y_te[test_mask]
        if len(np.unique(y_tr_g)) < 2 or len(np.unique(y_te_g)) < 2:
            continue
        p_g, s_g = fit_probe(X_tr_g, y_tr_g)
        m_g = evaluate_probe(p_g, s_g, X_te_g, y_te_g)
        cross_results[held_out] = m_g
        print(f"[{model_key}] held-out '{held_out}': AUROC={m_g['auroc']:.3f}")
    out["cross_strategy"] = cross_results

    # --- bootstrap CI on multi-layer ---------------------------------
    mean_a, lo, hi = bootstrap_auroc(multi_probs, y_te)
    out["multilayer"]["auroc_95ci"] = [lo, hi]

    # --- plots -------------------------------------------------------
    plot_layer_auroc(
        layer_aurocs, multi_metrics["auroc"],
        str(out_dir / "fig_layer_auroc.png"),
        title=f"Exp 2 — Layer-wise injection-probe AUROC ({model_id})",
    )
    plot_calibration(
        multi_probs, y_te,
        str(out_dir / "fig_calibration.png"),
        title=f"Exp 2 — Multi-layer probe calibration ({model_id})",
    )
    plot_confusion(
        multi_preds, y_te,
        str(out_dir / "fig_confusion.png"),
        labels=("Clean", "Injected"),
        title=f"Exp 2 — Confusion matrix ({model_id})",
    )

    # PCA at best single layer coloured by 4-way strategy
    strategy_to_int = {"clean": 0, "direct": 1, "indirect": 2, "obfuscated": 3}
    strat_int = np.array([strategy_to_int[s] for s in strategies])
    plot_pca_embedding(
        H[:, best_layer, :],
        strat_int,
        str(out_dir / "fig_pca.png"),
        label_names={0: "Clean", 1: "Direct override",
                     2: "Indirect contradiction", 3: "Obfuscated"},
        title=f"Exp 2 — PCA of hidden states, layer {best_layer} ({model_id})",
    )

    # Layer-wise table
    save_metrics_table(
        layer_metrics,
        ["layer", "auroc", "accuracy", "precision", "recall", "f1", "brier", "ece"],
        str(out_dir / "table_layer_metrics.csv"),
    )

    out["all_layer_metrics"] = layer_metrics
    return out, H, H_tr, H_te, y_tr, y_te


def cross_model_transfer(
    Ha: np.ndarray, ya: np.ndarray, name_a: str,
    Hb: np.ndarray, yb: np.ndarray, name_b: str,
    out_path: str,
) -> dict:
    """Test whether a probe trained on model A's hidden states transfers to
    model B by projecting both into a shared low-dimensional subspace via PCA.

    We average across the top-3 best layers of each model (selected by AUROC
    against y) and project to a 64-D shared subspace.
    """
    rng = np.random.default_rng(SEED)

    def _avg_top_layers(H: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
        # Score each layer by quick logistic AUROC, take top-k, average.
        n_layers = H.shape[1]
        layer_scores = []
        for L in range(n_layers):
            try:
                p, s = fit_probe(H[:, L, :], y)
                pr = probe_predict_proba(p, s, H[:, L, :])
                layer_scores.append(roc_auc_score(y, pr))
            except Exception:
                layer_scores.append(0.5)
        top = np.argsort(layer_scores)[::-1][:k]
        return H[:, top, :].mean(axis=1)

    Xa = _avg_top_layers(Ha, ya)
    Xb = _avg_top_layers(Hb, yb)

    # Reduce both to a shared 64-D PCA basis fit on A
    n_components = min(64, Xa.shape[0] - 1, Xa.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    Xa_red = pca.fit_transform(Xa)
    # Pad/truncate Xb to A's hidden dim before PCA-projecting (handle dim mismatch)
    if Xb.shape[1] != Xa.shape[1]:
        if Xb.shape[1] > Xa.shape[1]:
            Xb_aligned = Xb[:, :Xa.shape[1]]
        else:
            pad_w = Xa.shape[1] - Xb.shape[1]
            Xb_aligned = np.concatenate([Xb, np.zeros((Xb.shape[0], pad_w))], axis=1)
    else:
        Xb_aligned = Xb
    Xb_red = pca.transform(Xb_aligned)

    # Train probe on A, test on B (same labels both sides)
    p, s = fit_probe(Xa_red, ya)
    metrics_a_to_b = evaluate_probe(p, s, Xb_red, yb)

    # And the reverse direction
    p2, s2 = fit_probe(Xb_red, yb)
    metrics_b_to_a = evaluate_probe(p2, s2, Xa_red, ya)

    print(f"[transfer] {name_a} -> {name_b}: AUROC={metrics_a_to_b['auroc']:.3f}")
    print(f"[transfer] {name_b} -> {name_a}: AUROC={metrics_b_to_a['auroc']:.3f}")

    # Plot a small bar chart of the four AUROCs (A->A, A->B, B->A, B->B)
    p_aa, s_aa = fit_probe(Xa_red, ya)
    self_a = evaluate_probe(p_aa, s_aa, Xa_red, ya)["auroc"]
    p_bb, s_bb = fit_probe(Xb_red, yb)
    self_b = evaluate_probe(p_bb, s_bb, Xb_red, yb)["auroc"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = [
        (f"{name_a} → {name_a}", self_a, PALETTE["primary"]),
        (f"{name_a} → {name_b}", metrics_a_to_b["auroc"], PALETTE["warm"]),
        (f"{name_b} → {name_a}", metrics_b_to_a["auroc"], PALETTE["violet"]),
        (f"{name_b} → {name_b}", self_b, PALETTE["good"]),
    ]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars],
           edgecolor="white", linewidth=1.2)
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle=":", label="Random (0.50)")
    for i, (lbl, val, _c) in enumerate(bars):
        ax.text(i, val + 0.015, f"{val:.3f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Exp 2 — Cross-model probe transfer (shared PCA subspace)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "self_a": self_a,
        "self_b": self_b,
        f"{name_a}_to_{name_b}": metrics_a_to_b,
        f"{name_b}_to_{name_a}": metrics_b_to_a,
    }


def plot_combined_layer_curves(
    per_model: dict[str, dict], out_path: str
) -> None:
    """Layer-wise AUROC curve for both models on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["primary"], PALETTE["warm"]]
    for (key, summ), color in zip(per_model.items(), colors):
        layers = np.arange(len(summ["layer_aurocs"]))
        # Normalise x-axis to relative depth so two different models compare cleanly
        rel_depth = layers / (len(summ["layer_aurocs"]) - 1)
        ax.plot(rel_depth, summ["layer_aurocs"],
                color=color, marker="o", markersize=4,
                label=f"{summ['model']} (best={max(summ['layer_aurocs']):.3f})")
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle=":", label="Random (0.50)")
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("AUROC")
    ax.set_title("Exp 2 — Layer-wise injection-probe AUROC across gateway models")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_strategy_table(per_model: dict[str, dict], out_path: str) -> None:
    """Heatmap of cross-strategy generalisation AUROC for both models."""
    rows = list(per_model.keys())
    cols = ["direct", "indirect", "obfuscated"]
    data = np.zeros((len(rows), len(cols)))
    for i, k in enumerate(rows):
        for j, s in enumerate(cols):
            cs = per_model[k]["cross_strategy"].get(s)
            data[i, j] = cs["auroc"] if cs else np.nan

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlGnBu",
        xticklabels=[c.title() for c in cols],
        yticklabels=[per_model[k]["model"].split("/")[-1] for k in rows],
        cbar_kws={"label": "Held-out AUROC"},
        vmin=0.5, vmax=1.0, ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Held-out injection strategy")
    ax.set_ylabel("Gateway model")
    ax.set_title("Exp 2 — Cross-strategy probe generalisation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    print(f"[setup] device={DEVICE}  dtype={DTYPE}")

    texts, labels, strategies = build_experiment2_dataset()
    labels = np.array(labels)
    strategies = np.array(strategies)
    print(f"[data] N={len(texts)}  clean={int((labels==0).sum())}  "
          f"injected={int((labels==1).sum())}")

    per_model: dict[str, dict] = {}
    cached: dict[str, tuple] = {}

    for key, model_id in MODELS.items():
        try:
            summary, H, H_tr, H_te, y_tr, y_te = run_per_model(
                key, model_id, texts, labels, strategies
            )
            per_model[key] = summary
            cached[key] = (H, labels)
        except Exception as exc:
            print(f"[{key}] FAILED with {exc!r}")
            per_model[key] = {"model": model_id, "key": key, "error": repr(exc)}

    # Combined plots
    if all("layer_aurocs" in v for v in per_model.values()) and len(per_model) >= 2:
        plot_combined_layer_curves(per_model, str(RESULTS_DIR / "fig_layer_auroc_combined.png"))
        plot_strategy_table(per_model, str(RESULTS_DIR / "fig_cross_strategy_heatmap.png"))

    # Cross-model transfer
    transfer_summary: dict | None = None
    if len(cached) >= 2:
        keys = list(cached.keys())
        Ha, ya = cached[keys[0]]
        Hb, yb = cached[keys[1]]
        transfer_summary = cross_model_transfer(
            Ha, ya, keys[0],
            Hb, yb, keys[1],
            str(RESULTS_DIR / "fig_cross_model_transfer.png"),
        )

    # Save full summary JSON
    final = {
        "device": DEVICE,
        "n_samples": int(len(texts)),
        "models": MODELS,
        "per_model": {
            k: {kk: vv for kk, vv in v.items() if kk != "all_layer_metrics"}
            for k, v in per_model.items()
        },
        "cross_model_transfer": transfer_summary,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"[done] artifacts written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

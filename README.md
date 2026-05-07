# Reproducible White-Box Jailbreak Monitoring

Two probing experiments that test whether a small language model's internal hidden states encode reliable safety-relevant signals, without modifying the model at all.

The core question: if you freeze a model and just read out its internal activations as it processes different inputs, can a simple classifier trained on those activations detect uncertainty, adversarial injection, and novel attack strategies it has never seen?

---

## Background

Most LLM safety systems work by classifying each message in isolation. This has a known weakness: an attacker can craft inputs that individually look benign but cumulatively steer the model toward harmful output. A monitor that reads the model's internal representations, rather than just its text output, could catch signals the surface-level classifier misses.

This repo tests the foundation of that idea. Before building a full trajectory-based monitor, you need to verify that:

1. A model's hidden states actually separate uncertain inputs from clear ones (Experiment 1)
2. A model's hidden states actually separate adversarially injected inputs from clean ones (Experiment 2)

Both experiments use frozen models. No fine-tuning. No weight updates. The only thing being trained is a small logistic regression classifier on top of the extracted activations.

---

## Repository Structure

```
experiments/
    datasets.py                   dataset construction for both experiments
    probe_utils.py                shared utilities: extraction, probing, evaluation, plotting
    experiment1_uncertainty_probe.py
    experiment2_injection_probe.py
    requirements.txt

    results/
        experiment1/
            results.json          all metrics
            table_layer_metrics.csv
            fig_layer_auroc.png
            fig_baseline_comparison.png
            fig_calibration.png
            fig_confusion.png
            fig_pca.png

        experiment2/
            results.json
            fig_layer_auroc_combined.png
            fig_cross_strategy_heatmap.png
            fig_cross_model_transfer.png
            smollm2/              per-model figures and layer metrics
            qwen15b/
```

---

## Experiment 1: Uncertainty Probe

**Question.** Does a monitor model's hidden state encode a reliable "I am not sure" signal?

**Model.** Qwen/Qwen2.5-1.5B-Instruct (frozen, used as feature extractor only)

**Dataset.** 180 prompts across three equal buckets:
- Clearly safe: benign factual and creative requests
- Clearly unsafe: direct harmful requests
- Genuinely ambiguous: dual-use cybersecurity questions, contextual edge cases

The uncertainty label is binary. Clearly safe and clearly unsafe are labelled "certain". Ambiguous are labelled "uncertain". This is the signal a monitor would need in order to decide whether to abstain or escalate.

**Method.**
1. Pass each prompt through Qwen in a single forward pass
2. Extract the last token's hidden state at every transformer layer (29 layers total)
3. Train a logistic regression probe per layer on these 180 labelled examples
4. Train one additional probe that concatenates features from all layers
5. Train a TF-IDF plus logistic regression baseline on raw text as a control

The TF-IDF baseline is important. If the neural probe only matches TF-IDF performance, it means the signal is just topic detection, not anything geometric. The probe is only doing something meaningful if it clearly beats this baseline.

**Results.**

| Method | AUROC | F1 | Calibration Error |
|--------|-------|-----|-------------------|
| TF-IDF baseline | 0.926 | 0.857 | 0.325 |
| Best single layer (layer 7) | 1.000 | 0.971 | 0.021 |
| Multi-layer aggregated probe | 1.000 | 0.971 | 0.020 |

The neural probe beats TF-IDF by 7.4 AUROC points. The calibration error of 0.020 versus 0.325 for TF-IDF means the probe's confidence scores are actually meaningful for threshold-based abstention decisions, not just ranking. The uncertainty signal is strongest at layer 7, well before the final layer.

---

## Experiment 2: Injection Probe

**Question.** Can linear probes on a gateway model detect adversarial context injection, including attack strategies the probe has never seen?

**Models.**
- HuggingFaceTB/SmolLM2-1.7B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct

Both frozen.

**Dataset.** 200 samples across 50 base scenarios, each with four variants:
- Clean: no injection
- Direct: explicit instruction override ("ignore previous instructions")
- Indirect: subtle factual contradiction designed to steer output
- Obfuscated: encoded or rephrased injection to evade surface detection

Label is binary: clean is 0, any injection is 1.

**Method.**
1. Extract last-token hidden states across all layers for both models
2. Train per-layer and multi-layer probes on 140 training examples
3. Run cross-strategy generalisation test: train on 2 strategies, test on the held-out third
4. Run cross-model transfer test: train probe on one model's activations, apply to the other

The cross-strategy test is the most important evaluation. In deployment, an adversary will not use the exact attack types the probe was trained on. If AUROC stays high on the held-out strategy, it means the probe learned something general about adversarial structure, not just pattern matching specific wording.

**Results.**

In-distribution (both models):

| Metric | SmolLM2 | Qwen2.5-1.5B |
|--------|---------|--------------|
| AUROC | 1.000 | 1.000 |
| F1 | 1.000 | 1.000 |

Cross-strategy generalisation (held-out strategy AUROC):

| Held-out Strategy | SmolLM2 | Qwen2.5-1.5B |
|-------------------|---------|--------------|
| Direct | 0.945 | 0.958 |
| Indirect | 1.000 | 1.000 |
| Obfuscated | 1.000 | 1.000 |

A probe trained only on indirect and obfuscated examples still correctly ranked direct injection inputs it had never seen at AUROC 0.945. This confirms the probe captures a general adversarial signal in the hidden states.

Cross-model transfer:

| Direction | AUROC |
|-----------|-------|
| SmolLM2 to Qwen | 0.448 |
| Qwen to SmolLM2 | 0.420 |

Probe geometry does not transfer between models. Each model organises its internal representations differently. This is a genuine constraint worth knowing before building a monitoring system that assumes probes carry over across model updates.

---

## How to Run

**Install dependencies.**

```bash
pip install -r experiments/requirements.txt
```

A GPU is recommended. Both experiments run on CPU but will be slow. On GPU with CUDA, each experiment takes around 10 to 20 minutes depending on hardware.

**Run Experiment 1.**

```bash
cd experiments
python experiment1_uncertainty_probe.py
```

Results are written to `results/experiment1/`.

**Run Experiment 2.**

```bash
cd experiments
python experiment2_injection_probe.py
```

Results are written to `results/experiment2/`.

---

## How the Probe Works

The logistic regression probe is the only component being trained, on our labelled examples.

The underlying language model (Qwen, SmolLM2) is never modified. It acts purely as a feature extractor. Each prompt is passed through a single forward pass, and the last token's hidden state is read out at every transformer layer. These hidden states are just lists of numbers representing how the model has encoded that input internally.

The probe then learns a decision boundary in that numeric space using our labels. The reason this is interesting is that we never told the model anything about uncertainty or adversarial inputs. It learned its representations from general pretraining. Yet those representations organise themselves such that different input types end up in different regions. The probe just finds that boundary.

---

## Requirements

```
torch>=2.1
transformers>=4.41
accelerate>=0.30
sentencepiece>=0.2
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
einops>=0.7
```

---

## Notes

- If you have `torchao` installed, uninstall it before running. It breaks the transformers import chain on some setups: `pip uninstall torchao -y`
- Models are downloaded automatically from HuggingFace on first run
- All results in the `results/` folder are from runs on a CUDA GPU with the exact code in this repo

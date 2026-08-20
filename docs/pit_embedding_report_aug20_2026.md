# Embedding Quality Report: PIT-1B Base Model

## Summary

While building a point-in-time embedding pipeline for earnings call transcripts, we tested two model families: **PIT** (Diamegs) and **DatedGPT**. Both provide annual checkpoints trained on data up to a specific year, enabling look-ahead-free embeddings.

We found that the PIT-1B base model (December 2013 checkpoint) produces last-layer hidden states with a representation-scale issue that may require normalization before use in downstream financial applications. DatedGPT does not exhibit this issue to the same degree. We document the finding below and suggest next steps.

**Limitation:** This comparison uses one checkpoint from each model family, 200 transcripts, and one extraction method (last-layer mean pooling, 512-token truncation). The two models use different tokenizers (32k SentencePiece vs 50k BPE), so the same 512-token limit covers different amounts of source text. Neither model is a dedicated sentence encoder. The decisive test is downstream predictive performance (e.g., predicting abnormal returns), not embedding statistics alone.

## Test Setup

We embedded the **same 200 quarterly earnings call transcripts** (2014 Q1, US firms) using both models. Each transcript was truncated to 512 tokens and embedded via mean pooling over the last hidden layer. Statistics were computed in float32 after pooling.

## Results

| Metric | DatedGPT-2013-base | PIT-1B-201312 |
|--------|-------------------|---------------|
| Hidden dimension | 2,048 | 1,536 |
| Embedding mean | -0.008 | -0.079 |
| Embedding std | 0.80 | 14.62 |
| Max absolute value | 23.7 | 647.0 |
| Dims with per-dim abs max > 10 | 4 (0.2%) | 1,449 (94%) |
| L2 norm (mean) | 36.2 | 566.3 |

*Note: "Dims with per-dim abs max > 10" counts hidden dimensions whose maximum absolute value across all 200 documents exceeds 10.*

### The observation: high-magnitude dimensions in PIT

In the PIT-1B base model, two hidden dimensions carry large values with relatively low cross-document variation:

| Dimension | Mean | Std | Std / |Mean| |
|-----------|------|-----|---------------|
| 1531 | 406.5 | 76.0 | 0.19 |
| 1352 | -217.4 | 44.7 | 0.21 |

These dimensions account for a large share of the embedding vector's L2 norm. Their cross-document standard deviations (76 and 45) are not negligible in absolute terms, but are small relative to their means, suggesting that much of their magnitude is shared across documents rather than encoding document-specific information.

DatedGPT's largest dimension (mean = 20.1, std = 3.2) is 20x smaller in magnitude.

### Potential implications for downstream use

Without normalization, these high-magnitude dimensions could:

1. **Distort distance and similarity metrics** (cosine similarity, L2 distance), since a large share of the vector's direction is determined by dimensions that vary relatively little across documents
2. **Affect regression-based models**, depending on the scaling and regularization approach used
3. **Concentrate scale** in a small number of dimensions, though a formal effective-dimensionality analysis (e.g., PCA participation ratio) would be needed to quantify this

These are potential concerns, not confirmed effects. Whether they materially impact a specific downstream task (e.g., predicting post-earnings abnormal returns) depends on the task, the model specification, and any preprocessing applied.

## Possible Explanations

The transformer literature has documented similar high-magnitude activations in certain hidden dimensions (sometimes called "massive activations" or "outlier features"). Possible contributing factors include:

- Differences in network depth (PIT-1B has 52 layers, DatedGPT has 24), which may allow more accumulation through residual connections, though depth alone does not establish causality
- Differences in initialization, normalization placement, or training dynamics between the two architectures
- The language modeling objective does not directly constrain hidden-state magnitudes, since the LM head projection can in principle compensate (this remains speculative)

A rigorous analysis would require layer-by-layer activation measurements and controlled ablation studies.

## Suggested Next Steps

| Approach | Notes |
|----------|-------|
| **Per-dimension standardization** | Zero mean, unit variance per dim. Fit on training sample only, apply identically to test. Does not by itself fix cosine similarity without subsequent L2 normalization. |
| **PIT-4B-FT** (fine-tuned variant) | May have better-behaved hidden states. To be tested with the same diagnostic. |
| **Intermediate-layer embeddings** | Extract from an earlier layer (e.g., layer 26) where the scale concentration may be less pronounced. Exploratory. |
| **Remove or residualize outlier dimensions** | Drop or regress out dimensions with high mean-to-std ratio. Ad hoc, but useful as an exploratory diagnostic. |
| **Architectural investigation** | Layer-by-layer analysis to identify where the scale concentration emerges. Could inform future model releases. |
| **Downstream validation** | The most important test. Run both models' embeddings through the actual prediction task (e.g., PEAD regression) and compare out-of-sample performance. Embedding statistics are suggestive but not conclusive. |

## Appendix: Embedding Statistics

### One example transcript, embedded by both models

```
PIT-1B-201312:   mean=-0.062, std=15.08, min=-226.7, max=439.1, abs_max_dim=1531
DatedGPT-2013:   mean=-0.008, std=0.80,  min=-20.3,  max=23.7,  abs_max_dim=2030
```

PIT's max (439) is 18x larger than DatedGPT's (24), driven by dimension 1531.

### Top dimensions in PIT-1B-201312 (across 200 earnings calls)

```
dim 1531: abs_max=647.0, mean=406.5, std=76.0  (std/|mean| = 0.19)
dim 1352: abs_max=379.0, mean=-217.4, std=44.7  (std/|mean| = 0.21)
dim 1081: abs_max=139.0, mean=21.9,  std=17.8  (std/|mean| = 0.81)
```

### Top dimensions in DatedGPT-2013-base (across 200 earnings calls)

```
dim 2030: abs_max=23.7, mean=20.1, std=3.2  (std/|mean| = 0.16)
dim 1073: abs_max=20.3, mean=-18.1, std=2.7  (std/|mean| = 0.15)
dim 2006: abs_max=12.9, mean=-10.9, std=2.1  (std/|mean| = 0.19)
```

DatedGPT also has dimensions with low relative variation, but their absolute magnitudes are 20x smaller, so they do not dominate the overall embedding geometry to the same degree.

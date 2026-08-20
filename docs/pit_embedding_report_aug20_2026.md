# Embedding Quality Report: PIT-1B Base Model

## Summary

While building a point-in-time embedding pipeline for earnings call transcripts, we tested two model families: **PIT** (Diamegs) and **DatedGPT**. Both provide annual checkpoints trained on data up to a specific year, enabling look-ahead-free embeddings.

We found that the PIT-1B base model produces embeddings with a structural quality issue that makes them unsuitable for direct use in financial applications. DatedGPT does not exhibit this issue. We document the finding below and suggest potential remedies.

## Test Setup

We embedded the **same 200 quarterly earnings call transcripts** (2014 Q1, US firms) using both models. Each transcript was truncated to 512 tokens and embedded via mean pooling over the last hidden layer.

## Results

| Metric | DatedGPT-2013-base | PIT-1B-201312 |
|--------|-------------------|---------------|
| Hidden dimension | 2,048 | 1,536 |
| Embedding mean | -0.008 | -0.079 |
| Embedding std | 0.80 | 14.62 |
| Max absolute value | 23.7 | **647.0** |
| Dimensions with abs > 10 | 4 (0.2%) | **1,449 (94%)** |
| L2 norm (mean) | 36.2 | **566.3** |

### The issue: dominant outlier dimensions

In the PIT-1B base model, **two hidden dimensions carry extremely large, nearly constant values across all documents**:

- Dimension 1531: mean = 406.5, std = 76.0
- Dimension 1352: mean = -217.4, std = 44.7

These dimensions account for a disproportionate share of the embedding vector's magnitude. Because their variance is low relative to their mean (std/mean ratio of ~0.19), they carry very little information about differences between transcripts. In effect, they act as large bias terms that overshadow the meaningful variation in other dimensions.

### Why this matters for financial applications

When using embeddings as features (e.g., predicting abnormal returns, constructing text-based surprise measures), the outlier dimensions would:

1. **Dominate any distance or similarity metric** (cosine similarity, L2 distance), making all documents appear nearly identical
2. **Overwhelm regression coefficients**, since a regularized model would allocate most weight to explaining the large-magnitude dimensions rather than the informative ones
3. **Reduce the effective dimensionality** of the representation from 1,536 to a much smaller number

DatedGPT's embeddings do not have this problem. Its largest dimension (mean = 20.1) is 20x smaller than PIT's, and 99.8% of dimensions stay below an absolute value of 10.

## Possible Explanations

This phenomenon has been documented in the transformer literature under terms like "massive activations" or "outlier features." Contributing factors may include:

- The depth of the network (PIT-1B has 52 layers vs DatedGPT's 24), which allows more accumulation through residual connections
- Differences in initialization or normalization placement between the architectures
- The language modeling objective does not directly constrain hidden-state magnitudes, since the LM head projection can compensate

We have not done a layer-by-layer analysis to pinpoint the exact cause, and this would be a worthwhile investigation for the PIT team.

## Potential Remedies

| Approach | Feasibility | Effectiveness |
|----------|-------------|---------------|
| **Per-dimension standardization** (zero mean, unit variance) | Easy, post-hoc | Good, but may amplify noise in low-variance dims |
| **Use PIT-4B-FT** (fine-tuned variant) | Easy if available | Reportedly better, but needs verification |
| **Intermediate-layer embeddings** (e.g., layer 26 instead of 51) | Easy to test | Outliers may be less severe in earlier layers |
| **Architectural fix** (e.g., layer norm after last block, hidden-state regularization during training) | Requires retraining | Would resolve the root cause |
| **Use DatedGPT instead** | Immediate | Avoids the issue entirely |

## Appendix: Raw Embedding Statistics

### DatedGPT-2013-base, 10 random samples from 200 earnings calls

```
Sample [4216]: mean=-0.062, std=15.08, min=-226.7, max=439.1  <- PIT
Sample [4216]: mean=-0.008, std=0.80,  min=-20.3,  max=23.7   <- DatedGPT (typical)
```

*The PIT sample's max (439) is 18x larger than DatedGPT's max (23.7), and both come from the same transcript.*

### Top outlier dimensions in PIT-1B-201312 (200 earnings calls)

```
dim 1531: abs_max=647.0, mean=406.5, std=76.0  (std/mean = 0.19)
dim 1352: abs_max=379.0, mean=-217.4, std=44.7  (std/mean = 0.21)
dim 1081: abs_max=139.0, mean=21.9,  std=17.8  (std/mean = 0.81)
```

Dimensions 1531 and 1352 are nearly constant across documents (low std/mean ratio), confirming they carry minimal discriminative information.

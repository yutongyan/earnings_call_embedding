# Point-in-Time Embedding Models: DatedGPT vs PIT

Comparison of two point-in-time language models for earnings call embeddings.

## Models

| | DatedGPT | PIT |
|--|----------|-----|
| Organization | datedgpt | Diamegs |
| Sizes | 1B | 1B, 4B |
| Variants | base, instruct | base, FT (fine-tuned) |
| Checkpoints | Annual (2013-2024) | Annual Dec (201312-202412) |
| Architecture | GPT-2 style | Custom GPT (PITForCausalLM) |
| Hidden dim | 2048 | 1536 (1B), 4096 (4B) |
| Layers | unknown | 52 (1B) |
| Training data | FineWeb monthly snapshots | FineWeb monthly snapshots |
| HuggingFace | datedgpt/datedgpt-{year}-base | Diamegs/PIT-{size}-{date} |

Both are trained on chronologically-ordered snapshots of FineWeb, where each checkpoint captures text available up to a specific date. This makes them suitable for point-in-time financial NLP where look-ahead bias must be avoided.

## Embedding Quality Comparison

Tested by extracting last hidden state with mean pooling over token positions.

### DatedGPT-2014-base (2048 dims)

```
Shape: (3, 12, 2048)
Overall: mean=0.00, std=1.22, min=-26.0, max=29.6
Per-dim abs max: max=22.4
Dims with abs > 100: 0
```

Well-behaved: mean near zero, standard deviation ~1, no outlier dimensions. Embeddings are directly usable for downstream tasks without normalization.

### PIT-1B-201312 base (1536 dims)

```
Shape: (8889, 1536)
Overall: mean=-0.07, std=13.74, min=-301.8, max=571.3
Per-dim abs max: max=571.3
Dims with abs > 100: many
```

Severe outlier dimension problem. Dimension 1531 dominates every sample with values 300-570. This dimension is nearly constant across all documents (std ~40 relative to mean ~400), carrying no discriminative information.

### PIT-1B-FT (fine-tuned) (1536 dims)

Reported to have better-behaved embeddings than the base model. Fine-tuning redistributes hidden state activations more evenly.

## Root Cause: Outlier Dimensions in Base Causal LMs

The PIT base model exhibits "massive activations" in specific hidden dimensions. This is a well-documented phenomenon in transformer language models:

1. **Residual stream accumulation**: With 52 layers and residual connections (`h = h + layer_output`), certain dimensions accumulate signal across all layers, growing roughly linearly with depth.

2. **Attention sink effect**: In causal LMs, early tokens receive disproportionate attention. Certain dimensions encode this positional/structural information rather than semantic content.

3. **No penalty during pre-training**: The language modeling objective (next-token prediction) only sees the output after the LM head projection, which can compensate for any hidden-state scale. There is no explicit regularization on hidden state magnitudes.

4. **Mean pooling amplifies the problem**: When we average over all token positions, a dimension that is consistently large at every position produces a massive pooled value. For next-token prediction this doesn't matter, but for embeddings it dominates the representation.

DatedGPT avoids this, likely due to architectural differences (fewer layers, different initialization, or implicit regularization in training).

## Practical Impact on Embeddings

| Issue | DatedGPT | PIT base |
|-------|----------|----------|
| Usable without normalization | Yes | No |
| Cosine similarity meaningful | Yes | No (dominated by outlier dim) |
| L2 distance meaningful | Yes | No |
| PCA/dimensionality reduction | Works directly | Must remove outliers first |
| As features in regression | Works directly | Outlier dim dominates coefficients |

## Compatibility with HuggingFace

| Feature | DatedGPT | PIT |
|---------|----------|-----|
| `output_hidden_states=True` | Works natively | Returns None (custom architecture) |
| `AutoModelForCausalLM` | Standard loading | Requires `trust_remote_code=True` |
| Forward hook needed | No | Yes (for base model) |
| `torch_dtype` | Deprecated, use `dtype` | Same |

## Recommendation

**Use DatedGPT-base** for earnings call embeddings:
- Clean hidden states, no post-processing needed
- Standard HuggingFace API compatibility
- Same point-in-time discipline (annual checkpoints 2013-2024)
- 2048-dim embeddings (vs 1536 for PIT-1B)

If PIT is required (e.g., for comparison or the 4B model), use the **FT (fine-tuned) variant** to mitigate the outlier dimension problem, or apply per-dimension standardization before downstream use.

## Model Assignment for Earnings Calls

For a transcript with call date in year Y, use the model trained on data up to year Y-1:

| Call Year | DatedGPT Model | PIT Model |
|-----------|---------------|-----------|
| 2014 | datedgpt-2013-base | PIT-1B-201312 |
| 2015 | datedgpt-2014-base | PIT-1B-201412 |
| 2016 | datedgpt-2015-base | PIT-1B-201511 |
| ... | ... | ... |
| 2019 | datedgpt-2018-base | PIT-1B-201812 |

Events before 2014 cannot be embedded (no model with cutoff before 2013).

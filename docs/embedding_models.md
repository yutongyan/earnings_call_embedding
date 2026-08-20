# Point-in-Time Embedding Models: DatedGPT vs PIT

Comparison of two point-in-time language models for earnings call embeddings.

## Models

| | DatedGPT | PIT |
|--|----------|-----|
| Organization | datedgpt | Diamegs |
| Sizes | 1B | 1B, 4B |
| Variants | base, instruct | base, FT (4B only) |
| Checkpoints | Annual (2013-2024) | Annual Dec (201312-202412) |
| Architecture | LlamaForCausalLM | Custom GPT (PITForCausalLM) |
| Hidden dim | 2048 | 1536 (1B), 4096 (4B) |
| Layers | 24 | 52 (1B), 20 (4B) |
| Vocab / Tokenizer | 32k SentencePiece | 50,304 GPT-2 BPE |
| Normalization | RMSNorm | RMSNorm |
| Position encoding | RoPE | RoPE |
| Training data | FineWeb monthly snapshots | FineWeb monthly snapshots |
| HuggingFace | datedgpt/datedgpt-{year}-base | Diamegs/PIT-{size}-{date} |

Both are trained on chronologically-ordered snapshots of FineWeb, where each checkpoint captures text available up to a specific date. This makes them suitable for point-in-time financial NLP where look-ahead bias must be avoided.

## Embedding Extraction Method

Both models are causal LMs (not sentence encoders). We extract embeddings by:

1. Tokenize the transcript text (truncated to max 512 tokens)
2. Run a forward pass through the model
3. Take the **last hidden state** (before the LM head projection)
4. **Mean pool** over all token positions, weighted by the attention mask (excludes padding tokens, includes BOS/EOS)
5. Convert from bfloat16 to float32 for statistics and storage

**DatedGPT**: `output_hidden_states=True` works natively via HuggingFace API. Use `outputs.hidden_states[-1]`.

**PIT**: `output_hidden_states=True` returns `None` due to the custom `PITForCausalLM` architecture. Requires a **forward hook** on the last transformer block (`transformer.h.51` for PIT-1B) to capture the hidden state. The block returns a tuple; take the first element.

**Tokenizer note**: Because the tokenizers differ (32k SentencePiece vs 50k BPE), a 512-token truncation covers different amounts of transcript text. DatedGPT's SentencePiece typically produces fewer tokens for the same text, so 512 tokens covers more content. A character-level or word-level truncation would be a fairer comparison.

## Embedding Quality Comparison

Tested on the **same 200 earnings call transcripts** (2014 Q1 calls, max 512 tokens, mean-pooled over token positions). Statistics computed in float32 after pooling.

### DatedGPT-2013-base (2048 dims)

```
Shape: (200, 2048)
Overall: mean=-0.008, std=0.80, min=-20.3, max=23.7
Per-dim abs max: mean=1.02, max=23.7
Dims > 100: 0, Dims > 50: 0, Dims > 10: 4
L2 norms: mean=36.2, std=1.6
Top outlier dims:
  dim 2030: abs_max=23.74, mean=20.11, std=3.23
  dim 1073: abs_max=20.31, mean=-18.11, std=2.73
  dim 2006: abs_max=12.95, mean=-10.94, std=2.12
```

Well-behaved overall. Three mild outlier dims with means 10-20, but 20x smaller than PIT's outliers. 99.8% of dimensions have abs_max < 10. L2 norms are compact (mean=36) and consistent (std=1.6).

### PIT-1B-201312 base (1536 dims)

```
Shape: (200, 1536)
Overall: mean=-0.079, std=14.62, min=-379.0, max=647.0
Per-dim abs max: mean=18.08, max=647.0
Dims > 100: 5, Dims > 50: 19, Dims > 10: 1,449 (94% of dims)
L2 norms: mean=566.3, std=87.8
Top outlier dims:
  dim 1531: abs_max=647.0, mean=406.5, std=76.0
  dim 1352: abs_max=379.0, mean=-217.4, std=44.7
  dim 1081: abs_max=139.0, mean=21.9, std=17.8
```

Severe outlier dimension problem. Two dimensions (1531, 1352) dominate every sample with means 406 and -217. 94% of all dimensions exceed abs_max of 10. L2 norms are 16x larger and 56x more variable than DatedGPT.

## Outlier Dimensions: Observed Phenomenon

The PIT base model exhibits extremely large activations in specific hidden dimensions. This is an empirical observation documented in the transformer literature (sometimes called "massive activations" or "kurtosis outliers").

**Observed facts:**
- Two PIT-1B dimensions (1531, 1352) have mean absolute values of 400+ and 217+ across all samples
- These dimensions have low variance relative to their magnitude (std/mean ratio ~0.19), meaning they are nearly constant and carry minimal discriminative information between documents
- DatedGPT has mild outlier dims (mean ~20) but 20x smaller in magnitude

**Possible contributing factors** (hypotheses, not established causes):
- Residual stream dynamics across 52 layers (PIT) vs 24 layers (DatedGPT), where certain dimensions may accumulate signal through residual connections
- Differences in initialization, normalization placement, or training dynamics between the two architectures
- The LM objective optimizes predictions through the LM head, which can absorb hidden-state scale differences, leaving certain dimensions free to grow without affecting loss

A rigorous analysis would require layer-by-layer activation measurements and ablation studies, which is beyond the scope of this comparison.

## Practical Impact on Embeddings

| Issue | DatedGPT | PIT base |
|-------|----------|----------|
| Usable with L2 normalization | Yes | Partially (outlier dims dominate direction) |
| Cosine similarity meaningful | Likely (needs downstream validation) | No (dominated by outlier dims) |
| PCA/dimensionality reduction | Works directly | Must standardize per-dim first |
| As features in regression | Reasonable | Outlier dims dominate coefficients |

**Important caveat**: Neither model is a dedicated sentence encoder. Downstream predictive performance (e.g., predicting abnormal returns) is the ultimate validation, not embedding statistics alone. Both models should be evaluated on the actual PEAD task before drawing conclusions.

## Compatibility with HuggingFace

| Feature | DatedGPT | PIT |
|---------|----------|-----|
| `output_hidden_states=True` | Works natively | Returns None (custom architecture) |
| `AutoModelForCausalLM` | Standard loading | Requires `trust_remote_code=True` |
| Forward hook needed | No | Yes (on `transformer.h.{last}`) |

## Recommendation

**Use DatedGPT-base** for initial experiments:
- Cleaner hidden states, minimal post-processing needed
- Standard HuggingFace API compatibility
- Same point-in-time discipline (annual checkpoints 2013-2024)
- 2048-dim embeddings

**If PIT is required** (e.g., for the 4B model):
- Use the **PIT-4B-FT** (fine-tuned) variant if available. Note: PIT-1B-FT does not appear to exist in the public model listings. Only PIT-4B-FT checkpoints are published.
- Apply **per-dimension standardization** (zero mean, unit variance) before downstream use. Fit standardization parameters only on the training sample to avoid look-ahead bias.
- Consider evaluating intermediate-layer representations (not just the last layer), as outlier dimensions may be less severe in earlier layers.

## Model Assignment for Earnings Calls

For a transcript with call date in year Y, use the model trained on data up to year Y-1:

| Call Year | DatedGPT Model | PIT Model |
|-----------|---------------|-----------|
| 2014 | datedgpt-2013-base | PIT-1B-201312 |
| 2015 | datedgpt-2014-base | PIT-1B-201412 |
| 2016 | datedgpt-2015-base | PIT-1B-201511 |
| 2017 | datedgpt-2016-base | PIT-1B-201612 |
| 2018 | datedgpt-2017-base | PIT-1B-201712 |
| 2019 | datedgpt-2018-base | PIT-1B-201812 |

Events before 2014 cannot be embedded (no model with cutoff before 2013).

This annual assignment is a conservative approximation. For exact point-in-time discipline, compare the call timestamp with the model training cutoff timestamp, not only calendar years. For example, a call on January 2, 2014 could arguably use the 2013 model (trained through Dec 2013), which is what the table above specifies.

## Normalization Approaches

If post-processing is needed (especially for PIT):

| Method | Pros | Cons |
|--------|------|------|
| L2 normalization | Removes magnitude differences | PIT outlier dims still dominate direction |
| Per-dim standardization | Equalizes all dimensions | Can amplify low-variance noise; must fit on training data only |
| PCA after standardization | Removes correlated/uninformative dims | Adds complexity; fit on training data only |
| Remove outlier dims | Simple, targeted | Ad hoc threshold choice |
| Use intermediate layer | May have fewer outliers | Less semantic content than final layer |

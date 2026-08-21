#!/usr/bin/env python3
"""Test embedding quality across ALL PIT-1B checkpoints (201312-201812).

For each checkpoint, embed the same 200 transcripts and report stats.
Checks if the outlier dimension issue is persistent across checkpoints.

Usage:
    python3 -u src/pit_all_checkpoints_test.py
"""

import gc
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

PIT_CHECKPOINTS = ["201412", "201511", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412"]

def mean_pool(hidden, mask):
    mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)


def test_one_checkpoint(model_name, texts, device="cuda", batch_size=8, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(device).eval()

    # Find last transformer block for hook
    import re
    captured = {}
    blocks = [(n, m) for n, m in model.named_modules() if re.match(r"transformer\.h\.\d+$", n)]
    if blocks:
        def hook_fn(mod, inp, out):
            captured["h"] = out[0] if isinstance(out, tuple) else out
        hook = blocks[-1][1].register_forward_hook(hook_fn)
        hook_name = blocks[-1][0]
    else:
        hook = None
        hook_name = "none"

    all_emb = []
    all_emb_raw = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            captured.clear()
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            if "h" in captured:
                hidden = captured["h"]
            else:
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
                hidden = out.hidden_states[-1] if out.hidden_states else out.logits
            # Test both: with and without RMS norm
            hidden_normed = torch.nn.functional.rms_norm(hidden, (hidden.size(-1),))
            emb_raw = mean_pool(hidden, enc["attention_mask"])
            emb_normed = mean_pool(hidden_normed, enc["attention_mask"])
            all_emb.append(emb_normed.float().cpu().numpy())
            all_emb_raw.append(emb_raw.float().cpu().numpy())

    if hook:
        hook.remove()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(all_emb), np.vstack(all_emb_raw), hook_name


def main():
    print("Loading 200 test transcripts...", flush=True)
    df = pd.read_parquet("data/transcripts_2014.parquet", columns=["event_id", "presentation_text"])
    df = df.head(200)
    texts = [t[:2000] if isinstance(t, str) else "" for t in df["presentation_text"]]
    print(f"{len(texts)} transcripts loaded\n")

    results = []

    for ckpt in PIT_CHECKPOINTS:
        model_name = f"Diamegs/PIT-1B-{ckpt}"
        print(f"=== {model_name} ===", flush=True)

        emb, emb_raw, hook_name = test_one_checkpoint(model_name, texts)

        for label, e in [("WITH rms_norm", emb), ("WITHOUT rms_norm", emb_raw)]:
            dim_abs_max = np.abs(e).max(axis=0)
            norms = np.linalg.norm(e, axis=1)
            top_dim = np.argmax(dim_abs_max)

            r = {
                "checkpoint": ckpt,
                "norm_applied": label,
                "shape": str(e.shape),
                "mean": float(e.mean()),
                "std": float(e.std()),
                "max": float(e.max()),
                "dims_gt_10": int((dim_abs_max > 10).sum()),
                "dims_gt_100": int((dim_abs_max > 100).sum()),
                "l2_norm_mean": float(norms.mean()),
                "top_dim": int(top_dim),
                "top_dim_abs_max": float(dim_abs_max[top_dim]),
                "top_dim_mean": float(e[:, top_dim].mean()),
                "top_dim_std": float(e[:, top_dim].std()),
            }
            results.append(r)

            print(f"  {label}: std={r['std']:.4f}, max={r['max']:.1f}, dims>10={r['dims_gt_10']}, "
                  f"L2={r['l2_norm_mean']:.1f}, top_dim_mean={r['top_dim_mean']:.1f}")
        print(flush=True)

    # Summary table
    print("\n=== Summary: WITH vs WITHOUT rms_norm ===")
    print(f"{'Checkpoint':<12} {'Norm':<18} {'Dims>10':>8} {'Max':>8} {'L2Norm':>8} {'TopMean':>9}")
    for r in results:
        print(f"{r['checkpoint']:<12} {r['norm_applied']:<18} {r['dims_gt_10']:>8} {r['max']:>8.1f} {r['l2_norm_mean']:>8.1f} {r['top_dim_mean']:>9.1f}")

    pd.DataFrame(results).to_csv("data/pit_checkpoint_quality.csv", index=False)
    print("\nSaved data/pit_checkpoint_quality.csv")


if __name__ == "__main__":
    main()

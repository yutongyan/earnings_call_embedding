#!/usr/bin/env python3
"""Embed earnings call transcripts using point-in-time PIT-4B models.

Uses Diamegs/PIT-4B-YYYYMM models where YYYYMM is the training data cutoff.
For each transcript, selects the PIT model whose cutoff is strictly before
the earnings call date to avoid look-ahead bias.

Usage:
    python3 src/embed_pit.py --model_date 201312 --output data/embeddings/
    python3 src/embed_pit.py --all --output data/embeddings/
"""

import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PIT_MODELS = [
    "201312", "201412", "201511", "201612",
    "201712", "201812", "201912", "202012",
    "202112", "202212", "202312", "202412",
]


def get_pit_model_for_date(call_date):
    """Return the latest PIT model whose cutoff is strictly before call_date."""
    call_ym = call_date.year * 100 + call_date.month
    best = None
    for m in PIT_MODELS:
        m_int = int(m)
        if m_int < call_ym:
            best = m
    return best


def assign_transcripts_to_models(meta):
    """Assign each transcript to its appropriate PIT model."""
    meta = meta.copy()
    meta["pit_model"] = meta["call_date"].apply(get_pit_model_for_date)
    return meta


def mean_pool(hidden_states, attention_mask):
    """Mean pooling over token embeddings, respecting the attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_texts(texts, model_name, batch_size=4, max_length=512, device="cuda"):
    """Embed texts using a causal LM. Uses last hidden state with mean pooling."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    captured_hidden = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_hidden["last"] = output[0]
        else:
            captured_hidden["last"] = output

    hook = None
    last_block = None
    for name, module in model.named_modules():
        if "ln_f" in name or "final_layernorm" in name:
            last_block = (name, module)
            break
    if last_block is None:
        blocks = [(n, m) for n, m in model.named_modules() if re.match(r"transformer\.h\.\d+$", n)]
        if blocks:
            last_block = blocks[-1]
    if last_block is not None:
        hook = last_block[1].register_forward_hook(hook_fn)
        print(f"    Hook on: {last_block[0]}", flush=True)
    else:
        print("    WARNING: no suitable layer found for hook, will use logits", flush=True)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            captured_hidden.clear()
            model(input_ids=encoded["input_ids"],
                  attention_mask=encoded["attention_mask"])

            if "last" in captured_hidden:
                hidden = captured_hidden["last"]
            else:
                outputs = model(input_ids=encoded["input_ids"],
                               attention_mask=encoded["attention_mask"])
                hidden = outputs.logits
            embeddings = mean_pool(hidden, encoded["attention_mask"])
            all_embeddings.append(embeddings.float().cpu().numpy())

        if (i // batch_size) % 50 == 0:
            print(f"    Batch {i//batch_size+1}/{(len(texts)-1)//batch_size+1}", flush=True)

    if hook is not None:
        hook.remove()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(all_embeddings)


def load_transcript_text(event_ids, data_dir="data/"):
    """Load presentation + Q&A text for given event_ids."""
    eid_set = set(event_ids)
    texts = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "transcripts_20*.parquet"))):
        df = pd.read_parquet(f, columns=["event_id", "presentation_text", "qa_text"])
        matched = df[df["event_id"].isin(eid_set)].set_index("event_id")
        for eid, row in matched.iterrows():
            pres = row["presentation_text"] or ""
            qa = row["qa_text"] or ""
            texts[eid] = (pres + " " + qa).strip()
        del df, matched
        if len(texts) >= len(eid_set):
            break
    gc.collect()
    return texts


def embed_one_model(model_date, meta, output_dir, data_dir="data/",
                    batch_size=8, max_length=512, device="cuda"):
    """Embed all transcripts assigned to a specific PIT model."""
    model_meta = meta[meta["pit_model"] == model_date]
    if len(model_meta) == 0:
        print(f"  No transcripts for {model_date}", flush=True)
        return

    out_path = os.path.join(output_dir, f"embeddings_{model_date}.npz")
    if os.path.exists(out_path):
        print(f"  {model_date}: already exists, skipping", flush=True)
        return

    model_name = f"Diamegs/PIT-1B-{model_date}"
    print(f"  {model_date}: {len(model_meta):,} transcripts, model={model_name}", flush=True)

    event_ids = model_meta["event_id"].tolist()
    texts_dict = load_transcript_text(event_ids, data_dir)

    ordered_eids = [eid for eid in event_ids if eid in texts_dict]
    ordered_texts = [texts_dict[eid] for eid in ordered_eids]

    if not ordered_texts:
        print(f"  {model_date}: no texts found", flush=True)
        return

    print(f"  Embedding {len(ordered_texts):,} texts...", flush=True)
    embeddings = embed_texts(ordered_texts, model_name, batch_size, max_length, device)

    np.savez_compressed(
        out_path,
        event_ids=np.array(ordered_eids),
        embeddings=embeddings,
    )
    print(f"  Saved {out_path}: shape={embeddings.shape}", flush=True)

    del embeddings, ordered_texts, texts_dict
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_date", type=str, default=None,
                        help="Specific PIT model date (e.g., 201312)")
    parser.add_argument("--all", action="store_true",
                        help="Process all PIT models")
    parser.add_argument("--output", default="data/embeddings/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    meta = pd.read_parquet("data/merged_metadata.parquet",
                           columns=["event_id", "call_date"])
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    meta = assign_transcripts_to_models(meta)

    print(f"Total events: {len(meta):,}")
    print(f"Model assignment counts:")
    for m, count in meta["pit_model"].value_counts().sort_index().items():
        print(f"  {m}: {count:,}")
    print()

    no_model = meta["pit_model"].isna().sum()
    if no_model > 0:
        print(f"  {no_model:,} events before earliest PIT model (skipped)")
        meta = meta.dropna(subset=["pit_model"])

    if args.model_date:
        embed_one_model(args.model_date, meta, args.output,
                        batch_size=args.batch_size, max_length=args.max_length,
                        device=args.device)
    elif args.all:
        models_needed = sorted(meta["pit_model"].unique())
        print(f"Processing {len(models_needed)} PIT models\n")
        for model_date in models_needed:
            embed_one_model(model_date, meta, args.output,
                            batch_size=args.batch_size, max_length=args.max_length,
                            device=args.device)
    else:
        print("Specify --model_date or --all")


if __name__ == "__main__":
    main()

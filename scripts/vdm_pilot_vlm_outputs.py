"""VLM side of the VDM conditioning pilot: one Stage 0 checkpoint -> latents + captions.

For every pilot example (data/vdm_pilot/train_rows.json rows of Future-L1-50K, and
the TwiFF validation rows with extracted frames), this runs the Stage 0 VLM exactly
as in training -- context frame + question, then the assistant prefix
<|latent_start|><|latent_pad|>x4<|latent_end|> -- and saves:

  latents   final-layer hidden states at the 4 latent slots (4 x 2048), the
            conditioning signal for the pilot's "latent" arm (raw hidden states,
            not the Stage 0 latent head, so the bridge to Wan is learned from scratch)
  caption   the answer the same model then generates greedily after the latent span
            (--caption_tokens new tokens, <image>/<rimage> placeholders stripped),
            the conditioning text for the "caption" arm

Caption and latents come from the same forward context, so the caption-vs-latent
comparison in train_vdm_pilot.py is paired: same model, same input, two readouts.
--no_captions skips generation (latents only, ~10x faster), for a second checkpoint.

Output: <output_dir>/{train,val}_vlm.pt with {"keys": [...], "latents": (N,4,2048) bf16,
"captions": [...]}; keys are "<video>|<row index>".
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage0_latent_grounding import (  # noqa: E402
    LATENT_END,
    LATENT_PAD,
    LATENT_START,
    dtype_from_str,
    load_extracted_frames,
    load_model_and_tokenizer,
    log,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Stage 0 step_N LoRA checkpoint")
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_rows_json", required=True, help="data/vdm_pilot/train_rows.json")
    p.add_argument("--train_frames_root", required=True)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_frames_root", required=True)
    p.add_argument("--latent_span_len", type=int, default=4)
    p.add_argument("--caption_tokens", type=int, default=64)
    p.add_argument("--no_captions", action="store_true")
    p.add_argument("--splits", default="val,train")
    p.add_argument("--limit", type=int, default=None, help="Per-split cap, for quick tests")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def clean_caption(text):
    text = re.sub(r"<\|?r?image\|?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_rows(args, split):
    from datasets import load_from_disk

    if split == "train":
        ds = load_from_disk(args.train_metadata_dir)
        rows = json.load(open(args.train_rows_json))["rows"]
        frames_root = args.train_frames_root
    else:
        ds = load_from_disk(args.val_metadata_dir)
        frames_root = args.val_frames_root
        have = set(os.listdir(frames_root))
        rows = [i for i, v in enumerate(ds["video"]) if v[:-4] in have]
    if args.limit:
        rows = rows[:args.limit]
    return ds, rows, frames_root


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = dtype_from_str(args.dtype)
    load_args = argparse.Namespace(use_lora=True, resume_from=args.checkpoint, qwen_dir=args.qwen_dir)
    vlm, tokenizer, processor, _ = load_model_and_tokenizer(load_args, device, dtype)
    vlm.eval()
    pad_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)
    prefix = LATENT_START + LATENT_PAD * args.latent_span_len + LATENT_END

    for split in args.splits.split(","):
        out_path = os.path.join(args.output_dir, f"{split}_vlm.pt")
        state = torch.load(out_path) if os.path.exists(out_path) else {"keys": [], "latents": [], "captions": []}
        if isinstance(state["latents"], torch.Tensor):
            state["latents"] = list(state["latents"].unbind(0))
        done = set(state["keys"])
        ds, rows, frames_root = split_rows(args, split)
        log(f"{split}: {len(rows)} rows, {len(done)} already done")
        for n, i in enumerate(rows):
            row = ds[i]
            key = f"{row['video']}|{i}"
            if key in done:
                continue
            frames = load_extracted_frames(frames_root, row["video"], row["question_images_index"])
            messages = [
                {"role": "system", "content": "You are a helpful assistant that reasons about videos."},
                {"role": "user", "content": [{"type": "image"} for _ in frames]
                 + [{"type": "text", "text": row["question"]}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
            inputs = processor(text=[text], images=frames, return_tensors="pt").to(device)
            out = vlm(**inputs, output_hidden_states=True, return_dict=True)
            mask = inputs["input_ids"][0] == pad_id
            state["latents"].append(out.hidden_states[-1][0, mask, :].to(torch.bfloat16).cpu())
            caption = ""
            if not args.no_captions:
                gen = vlm.generate(**inputs, max_new_tokens=args.caption_tokens, do_sample=False)
                caption = clean_caption(tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:],
                                                         skip_special_tokens=True))
            state["captions"].append(caption)
            state["keys"].append(key)
            if (n + 1) % 50 == 0:
                log(f"  {split}: {n + 1}/{len(rows)}  e.g. {caption[:100]!r}")
            if (n + 1) % args.save_every == 0:
                torch.save({**state, "latents": torch.stack(state["latents"])}, out_path)
        torch.save({**state, "latents": torch.stack(state["latents"]), "checkpoint": args.checkpoint}, out_path)
        log(f"{split}: saved {len(state['keys'])} examples to {out_path}")


if __name__ == "__main__":
    main()

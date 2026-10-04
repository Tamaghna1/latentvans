"""Qwen's own hidden states over its generated answer, for the VDM pilot's "qwen" arm.

The caption arm sends the Stage 0 VLM's answer through a text bottleneck: decoded to
text, re-encoded by Wan's umT5. This script keeps the same information but skips the
bottleneck: it re-runs the same Stage 0 checkpoint on context frame + question + the
latent prefix + that answer (teacher-forced, exactly the text the caption arm encoded)
and saves the VLM's hidden states at the answer-token positions. train_vdm_pilot.py
--cond qwen then feeds them to Wan through a trained per-token projector.

Two layers are saved: the final layer and a middle one (--mid_layer), since final-layer
states are specialised for next-token prediction and intermediate layers often carry
more of the visual/semantic content.

Output: <vlm_dir>/{split}_qwen_states.pt with {"keys", "last": [(L_i, 2048) bf16],
"mid": [...], "mid_layer"}; keys match <vlm_dir>/{split}_vlm.pt.
"""
import argparse
import os
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
    p.add_argument("--checkpoint", required=True, help="the Stage 0 checkpoint that generated the captions")
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--vlm_dir", required=True, help="vdm_pilot_vlm_outputs.py output (captions + keys)")
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_frames_root", required=True)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_frames_root", required=True)
    p.add_argument("--latent_span_len", type=int, default=4)
    p.add_argument("--mid_layer", type=int, default=18, help="hidden_states index (0 = embeddings; Qwen2.5-VL-3B has 36 layers)")
    p.add_argument("--splits", default="val,train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    from datasets import load_from_disk

    device = torch.device("cuda")
    load_args = argparse.Namespace(use_lora=True, resume_from=args.checkpoint, qwen_dir=args.qwen_dir)
    vlm, tokenizer, processor, _ = load_model_and_tokenizer(load_args, device, dtype_from_str(args.dtype))
    vlm.eval()
    end_id = tokenizer.convert_tokens_to_ids(LATENT_END)
    prefix = LATENT_START + LATENT_PAD * args.latent_span_len + LATENT_END

    for split in args.splits.split(","):
        vlm_out = torch.load(os.path.join(args.vlm_dir, f"{split}_vlm.pt"))
        ds = load_from_disk(args.train_metadata_dir if split == "train" else args.val_metadata_dir)
        frames_root = args.train_frames_root if split == "train" else args.val_frames_root
        keys, captions = vlm_out["keys"], vlm_out["captions"]
        if args.limit:
            keys, captions = keys[:args.limit], captions[:args.limit]
        last, mid = [], []
        for n, (key, caption) in enumerate(zip(keys, captions)):
            row = ds[int(key.rsplit("|", 1)[1])]
            frames = load_extracted_frames(frames_root, row["video"], row["question_images_index"])
            messages = [
                {"role": "system", "content": "You are a helpful assistant that reasons about videos."},
                {"role": "user", "content": [{"type": "image"} for _ in frames]
                 + [{"type": "text", "text": row["question"]}]},
            ]
            text = (processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    + prefix + " " + caption)
            inputs = processor(text=[text], images=frames, return_tensors="pt").to(device)
            out = vlm(**inputs, output_hidden_states=True, return_dict=True)
            ids = inputs["input_ids"][0]
            start = int((ids == end_id).nonzero()[-1]) + 1  # first token after <|latent_end|>
            if start >= len(ids):
                start = len(ids) - 1  # empty caption: keep one state rather than none
            last.append(out.hidden_states[-1][0, start:].to(torch.bfloat16).cpu())
            mid.append(out.hidden_states[args.mid_layer][0, start:].to(torch.bfloat16).cpu())
            if (n + 1) % 200 == 0:
                log(f"  {split}: {n + 1}/{len(keys)} (answer tokens this example: {len(ids) - start})")
        out_path = os.path.join(args.vlm_dir, f"{split}_qwen_states.pt")
        torch.save({"keys": keys, "last": last, "mid": mid, "mid_layer": args.mid_layer,
                    "checkpoint": args.checkpoint}, out_path)
        log(f"{split}: saved {len(keys)} examples to {out_path} "
            f"(mean {sum(x.shape[0] for x in last) / max(len(last), 1):.1f} answer tokens)")


if __name__ == "__main__":
    main()

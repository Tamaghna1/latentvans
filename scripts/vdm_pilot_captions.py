"""Alternative captions for the VDM pilot's caption arm (what the text says, not how it is passed).

The first pilot's captions are the Stage 0 VLM's TwiFF-style reasoning ("In frame_1, the
man is seen ... By frame_2, ...") cut at 64 tokens: much of that budget describes the
context frame Wan already sees, and the future is often cut off. Three variants:

  oracle   the clip's own ground-truth annotation (meta_data Summary + Process) -- an upper
           bound on what a caption of this clip can tell Wan; no model involved (--mode oracle
           runs on CPU)
  full     the same Stage 0 VLM and prompt as before, but up to --max_new_tokens (192) tokens
  future   base Qwen2.5-VL-3B-Instruct (no Stage 0 adapter: the adapted model was fine-tuned to
           TwiFF's reasoning style) asked for one or two sentences about only what happens next

Output: <output_dir>/{split}_vlm.pt with {"keys", "captions"} in the same key order as
--keys_from (vdm_pilot_vlm_outputs.py output), which vdm_pilot_encode.py --what text reads.
Generation is batched (left padding) for speed.
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FUTURE_INSTRUCTION = ("\nWrite a caption for the video that comes next: in one or two sentences, describe only "
                      "what happens next -- the actions and changes after this frame -- not what is already visible.")


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["oracle", "full", "future"], required=True)
    p.add_argument("--keys_from", required=True, help="dir with {split}_vlm.pt whose keys/order to follow")
    p.add_argument("--checkpoint", help="Stage 0 checkpoint (mode full)")
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_frames_root", required=True)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_frames_root", required=True)
    p.add_argument("--max_new_tokens", type=int, default=None, help="default: 192 (full), 80 (future)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--latent_span_len", type=int, default=4)
    p.add_argument("--splits", default="val,train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def clean(text):
    text = re.sub(r"<\|?r?image\|?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def oracle_caption(row):
    meta = row.get("meta_data") or {}
    return clean(f"{meta.get('Summary', '')} {meta.get('Process', '')}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from datasets import load_from_disk

    vlm = processor = tokenizer = None
    if args.mode != "oracle":
        from train_stage0_latent_grounding import LATENT_END, LATENT_PAD, LATENT_START, load_extracted_frames
        device = torch.device("cuda")
        if args.mode == "full":
            from train_stage0_latent_grounding import load_model_and_tokenizer
            load_args = argparse.Namespace(use_lora=True, resume_from=args.checkpoint, qwen_dir=args.qwen_dir)
            vlm, tokenizer, processor, _ = load_model_and_tokenizer(load_args, device, torch.bfloat16)
            prefix = LATENT_START + LATENT_PAD * args.latent_span_len + LATENT_END
        else:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            processor = AutoProcessor.from_pretrained(args.qwen_dir)
            tokenizer = processor.tokenizer
            vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_dir, dtype=torch.bfloat16).to(device)
            prefix = ""
        vlm.eval()
        tokenizer.padding_side = "left"
        processor.tokenizer.padding_side = "left"
        max_new = args.max_new_tokens or (192 if args.mode == "full" else 80)

    for split in args.splits.split(","):
        ds = load_from_disk(args.train_metadata_dir if split == "train" else args.val_metadata_dir)
        frames_root = args.train_frames_root if split == "train" else args.val_frames_root
        keys = torch.load(os.path.join(args.keys_from, f"{split}_vlm.pt"))["keys"]
        if args.limit:
            keys = keys[:args.limit]
        rows = [ds[int(k.rsplit("|", 1)[1])] for k in keys]
        captions = []
        if args.mode == "oracle":
            captions = [oracle_caption(r) for r in rows]
        else:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                texts, images = [], []
                for r in batch:
                    frames = load_extracted_frames(frames_root, r["video"], r["question_images_index"])
                    q = r["question"] + (FUTURE_INSTRUCTION if args.mode == "future" else "")
                    msgs = []
                    if args.mode == "full":
                        msgs.append({"role": "system", "content": "You are a helpful assistant that reasons about videos."})
                    msgs.append({"role": "user", "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": q}]})
                    texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix)
                    images.extend(frames)
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(vlm.device)
                with torch.no_grad():
                    gen = vlm.generate(**inputs, max_new_tokens=max_new, do_sample=False)
                new = gen[:, inputs["input_ids"].shape[1]:]
                captions += [clean(tokenizer.decode(g, skip_special_tokens=True)) for g in new]
                if (start // args.batch_size) % 10 == 0:
                    log(f"  {split}: {start + len(batch)}/{len(rows)}  e.g. {captions[-1][:160]!r}")
        out = os.path.join(args.output_dir, f"{split}_vlm.pt")
        torch.save({"keys": keys, "captions": captions, "mode": args.mode}, out)
        mean_words = sum(len(c.split()) for c in captions) / max(len(captions), 1)
        log(f"{split}: saved {len(captions)} {args.mode} captions to {out} (mean {mean_words:.1f} words); "
            f"first: {captions[0][:200]!r}")


if __name__ == "__main__":
    main()

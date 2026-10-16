"""Visually grounded next-event captioning for the VANS VLM: do bounding boxes of the objects in the scene
give better next-event captions?

Grounding comes from scripts/ground_objects.py (Qwen2.5-VL-3B-Instruct boxes the objects, tools, ingredients
and hands in the last input frame at native resolution). All arms start from VANS's VLM LoRA and run
vans_coconut.py's training/eval loop unchanged (same rows, order, steps, lr, loss on the answer tokens), so
each compares directly with its `nocot` arm ("[Think][/Think][Ans]caption[/Ans]", ROUGE-L 0.276):

  ground_text   nocot + the last-frame object list with boxes appended to the user's instruction
  ground_zoom   ground_text + a zoomed crop of the active region (hands and what they touch) as an image
  ground_think  plain instruction; the model first writes, inside [Think], the objects the next step will
                use with their boxes in the last frame, then the caption. Targets: detected objects whose
                label shares a noun with the GT caption / next-step sentence. No detector at test time.
  blind         control: nocot with the video removed (instruction text only). How much of the caption
                metric a model reaches from the user's words and language priors alone.

  train: --mode train --arm ground_text|ground_zoom|ground_think --data_dir data/vans_train
  eval:  --mode eval  --arm ... --ckpt <dir> --bench_dir data/vans_eval
"""
import glob
import json
import os
import re

import torch
import torch.nn as nn

from vans_coconut import Coconut, evaluate, parse_args, train
from vans_rl_common import log

ARMS = ("ground_text", "ground_zoom", "ground_think", "blind")
# not object identities: colors, materials, sizes and filler words in detector labels and captions
NON_NOUNS = set("""a an the of with and in on at to from into onto by for its their his her is are be being
some other another one two three left right top bottom front back side small large big long short round
white black red green blue yellow brown grey gray silver golden gold orange pink purple beige dark light
wooden wood metal metallic stainless steel plastic glass ceramic clear transparent paper cardboard""".split())


def nouns(s):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", s.lower())} - NON_NOUNS


def fmt(objs):
    return "; ".join(f"{o['label']} [{', '.join(str(v) for v in o['box'])}]" for o in objs)


class Grounded(Coconut):
    def __init__(self, args, device, lora_init):
        super().__init__(args, device, lora_init)
        self.ground = {}
        for d in (args.data_dir, args.bench_dir):
            for f in sorted(glob.glob(os.path.join(d or "", "ground", "objects_shard*.jsonl"))) if d else []:
                for l in open(f):
                    r = json.loads(l)
                    r["dir"] = d
                    self.ground[r["sample_id"]] = r
        log(f"grounding loaded for {len(self.ground)} rows")
        self.blind = args.arm == "blind"
        self.checked = False

    def inputs(self, arm, row, train):
        """(instruction, zoom image path or None, [Think] text) for one row."""
        g = self.ground.get(row["sample_id"])
        objs = g["objects"] if g else []
        ins, img, think = row["instruction"], None, ""
        if arm in ("ground_text", "ground_zoom") and objs:
            ins += ("\nObjects in the last frame of the video (boxes [x1, y1, x2, y2] on a 0-1000 scale): "
                    + fmt(objs) + ".")
        if arm == "ground_zoom" and g:
            img = os.path.join(g["dir"], g["zoom"])
            ins += "\nThe image is a zoomed view of the active region of the last frame."
        if arm == "ground_think" and train:
            ref = nouns(row["gt_caption"] + " " + row.get("target_sentence", ""))
            keep = [o for o in objs if nouns(o["label"]) & ref]
            think = ("Objects the next step will use, in the last frame: " + fmt(keep) + "." if keep
                     else "No object the next step will use is visible in the last frame.")
        return ins, img, think

    def prompt(self, video_path, instruction, image=None):
        """MLLMAgent.tokenize's layout (system prompt; user: video, text) with an optional image between the
        video and the text. The first call checks it reproduces MLLMAgent.tokenize exactly without the image."""
        from qwen_vl_utils import process_vision_info
        proc = self.agent.tokenizer

        def make(img):
            content = [] if self.blind else [{"type": "video", "video": video_path}]
            content += [{"type": "image", "image": img}] if img else []
            content += [{"type": "text", "text": instruction}]
            msgs = [{"role": "system", "content": [{"type": "text", "text": proc.system_prompt}]},
                    {"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(msgs, self.device)
            inp = proc(text=[text], images=images, videos=None if videos is None else [videos], return_tensors="pt",
                       padding=True, padding_side="left").to(self.device)
            return {k: v for k, v in inp.items() if torch.is_tensor(v) and k != "labels"}

        out = make(image)
        if not self.checked and self.blind:
            log(f"blind prompt: {out['input_ids'].shape[1]} tokens, no video")
            self.checked = True
        if not self.checked:
            ref, mine = super().prompt(video_path, instruction), (make(None) if image else out)
            if not torch.equal(ref["input_ids"], mine["input_ids"]) or not torch.equal(
                    ref["pixel_values_videos"], mine["pixel_values_videos"]):
                raise RuntimeError("grounded prompt does not reproduce MLLMAgent.tokenize")
            log(f"prompt check ok ({mine['input_ids'].shape[1]} tokens without image, {out['input_ids'].shape[1]} with)")
            self.checked = True
        return out

    @torch.no_grad()
    def features(self, prompt):
        vf = self.video_features(prompt) if "pixel_values_videos" in prompt else None
        if "pixel_values" not in prompt:
            return vf, None
        return vf, self.base.visual(prompt["pixel_values"].to(self.base.visual.dtype), grid_thw=prompt["image_grid_thw"])

    def rope(self, ids, prompt):
        kw = {"second_per_grid_ts": prompt["second_per_grid_ts"]} if "second_per_grid_ts" in prompt else {}
        pos, _ = self.base.get_rope_index(ids, prompt.get("image_grid_thw"), prompt.get("video_grid_thw"),
                                          attention_mask=torch.ones_like(ids), **kw)
        return pos

    def build(self, prompt, vfeat, segments, ifeat=None):
        """Coconut.build with image features too (text segments only)."""
        input_ids = torch.cat([prompt["input_ids"][0]] + [x for _, x in segments])[None]
        emb = self.base.get_input_embeddings()(input_ids).float()
        for tok_id, feat in ((self.base.config.video_token_id, vfeat), (self.base.config.image_token_id, ifeat)):
            if feat is not None:
                emb = emb.masked_scatter((input_ids == tok_id)[..., None].expand_as(emb), feat.float())
        return emb, self.rope(input_ids, prompt), input_ids, []

    def loss(self, row, data_dir, arm, stage, c, max_stage, max_answer_tokens):
        ins, img, think = self.inputs(arm, row, train=True)
        prompt = self.prompt(os.path.join(data_dir, row["input_video"]), ins, img)
        vfeat, ifeat = self.features(prompt)
        tail = self.text_ids(f"{think}[/Think][Ans]{row['gt_caption']}[/Ans]<|im_end|>")[:max_answer_tokens]
        emb, pos, input_ids, _ = self.build(prompt, vfeat, [("ids", self.text_ids("[Think]")), ("ids", tail)], ifeat)
        start = input_ids.shape[1] - tail.shape[0]
        logits = self.logits(self.hidden(emb, pos)[:, start - 1:-1])
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids[0, start:]), 0

    @torch.no_grad()
    def answer(self, row, bench_dir, arm, c, max_stage, max_new_tokens):
        """Greedy "[Think]...[/Think][Ans]...[/Ans]" (recomputes the prefix each token, like Coconut.answer)."""
        ins, img, _ = self.inputs(arm, row, train=False)
        prompt = self.prompt(os.path.join(bench_dir, row["input_video"]), ins, img)
        vfeat, ifeat = self.features(prompt)
        emb, pos, input_ids, _ = self.build(prompt, vfeat, [("ids", self.text_ids("[Think]"))], ifeat)
        end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        out = []
        for _ in range(max_new_tokens):
            nxt = int(self.logits(self.hidden(emb, pos)[:, -1]).argmax(-1))
            if nxt == end_id:
                break
            out.append(nxt)
            input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            emb = torch.cat([emb, self.base.get_input_embeddings()(input_ids[:, -1:]).float()], dim=1)
            pos = self.rope(input_ids, prompt)
        return "[Think]" + self.tok.decode(out, skip_special_tokens=True)


if __name__ == "__main__":
    a = parse_args(ARMS)
    train(a, Grounded) if a.mode == "train" else evaluate(a, Grounded)

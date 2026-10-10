"""Diagnostic: native Qwen forward vs vans_coconut hand-built embeddings on the same tokens; hidden vs embedding scale."""
import json, os, sys, torch
sys.argv += ["--mode", "train", "--arm", "cot", "--output_dir", "/tmp/x"]
from vans_coconut import Coconut, parse_args
args = parse_args()
S = "/scratch/users/anirban/tamaghnam/latentvans"
m = Coconut(args, "cuda", os.path.join(args.vans_dir, "VANS_mllm.safetensors")); m.eval()
rows = [json.loads(l) for l in open(f"{S}/data/vans_train/benchmark.jsonl")][:4]
with torch.no_grad():
    for r in rows:
        prompt = m.prompt(os.path.join(f"{S}/data/vans_train", r["input_video"]), r["instruction"])
        vfeat = m.video_features(prompt)
        segs, lf, _ = m.layout("cot", r["think"], r["gt_caption"], 0, 2, 4)
        emb, pos, ids, _ = m.build(prompt, vfeat, segs)
        start = ids.shape[1] - sum(x.shape[0] for k, x in segs[lf:] if k == "ids")
        h = m.hidden(emb, pos)
        lo = m.logits(h[:, start - 1:-1]); tgt = ids[:, start:]
        ours = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1)).item()
        # native: same ids, HF merges video + computes rope itself
        kw = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.agent.mllm_backbone(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
        lo2 = out.logits[:, start - 1:-1].float()
        nat = torch.nn.functional.cross_entropy(lo2.reshape(-1, lo2.shape[-1]), tgt.reshape(-1)).item()
        # native with VANS-style text (single contiguous answer string)
        full, _, _ = m.agent.tokenize(m.agent.tokenizer, is_train=False, add_query=True, instructions=r["instruction"],
            caption=r["gt_caption"], thinking=r["think"], video=os.path.join(f"{S}/data/vans_train", r["input_video"]),
            device="cuda", torch_dtype=torch.bfloat16)
        full = {k: v for k, v in full.items() if torch.is_tensor(v) and k != "labels"}
        P = prompt["input_ids"].shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o3 = m.agent.mllm_backbone(**full)
        l3 = o3.logits[:, P - 1:-1].float(); t3 = full["input_ids"][:, P:]
        vans = torch.nn.functional.cross_entropy(l3.reshape(-1, l3.shape[-1]), t3.reshape(-1)).item()
        e = m.base.get_input_embeddings().weight
        sid = r["sample_id"]
        print(f"{sid}: ours={ours:.3f} native_same_ids={nat:.3f} native_vans_text={vans:.3f} "
              f"| hidden_rms={h[0, -50:].float().pow(2).mean().sqrt():.3f} emb_rms={e.float().pow(2).mean().sqrt():.4f} "
              f"| first answer tokens ours={m.tok.decode(tgt[0,:12])!r} vans={m.tok.decode(t3[0,:12])!r}", flush=True)

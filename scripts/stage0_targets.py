"""Stage 0 latent-grounding targets (v2), shared by training, evaluation and diagnostics.

v1 (train_stage0_latent_grounding.py's compute_future_embedding) regresses all
latent slots onto ONE vector: the mean over future frames of the mean over all
merged ViT patches. diagnose_stage0_grounding.py showed (job 61670, 2026-09-25)
that a model trained that way scored no better than "predict the average
target", and that this target was 96% cosine-similar to the same pooling of the
context frame. Part of that was the frame-index bug fixed in twiff_frames.py
(the "future" frame was ~0.1 s after the context frame); the rest is the target
design, which v2 changes along three independent axes:

  layout   "pooled"     one vector per example, repeated over every latent slot (v1)
           "quadrants"  one vector per 2x2 spatial quadrant of the merged patch grid,
                        averaged over future frames: slot i <- quadrant i, so the
                        latent span keeps coarse "where" information
  delta    False        target = future features (v1)
           True         target = future features - context features (what changes)
  stats    None         raw features (v1)
           path         z-score each target dim with mean/std estimated on training
                        examples (analyze_stage0_targets.py writes the file), so
                        predicting the average scores ~1.0 instead of looking small
                        just because a few high-magnitude dims dominate the MSE

Patch tokens come from the vision tower's pooler_output (post-merger, already
put back in raster order); v1 applied the merger to last_hidden_state, which is
in window order. That is harmless for a global mean but would scramble quadrants.
"""
import torch

QUADRANTS = 4


@torch.no_grad()
def merged_patch_grid(frame, processor, vision_tower, device, compute_dtype):
    """Merged (2048-dim) patch tokens of one frame as a (h, w, d) grid in raster order."""
    proc = processor(images=[frame], return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device=device, dtype=compute_dtype)
    grid_thw = proc["image_grid_thw"].to(device)
    out = vision_tower(pixel_values, grid_thw=grid_thw)
    merge = getattr(vision_tower, "spatial_merge_size", 2)
    t, h, w = (int(x) for x in grid_thw[0])
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        tokens = out.pooler_output
    else:
        # Older transformers: forward() returned merged, raster-ordered tokens directly.
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    tokens = tokens.float()
    hm, wm = h // merge, w // merge
    if tokens.shape[0] != t * hm * wm:
        raise RuntimeError(f"expected {t * hm * wm} merged tokens for grid {t}x{h}x{w}, got {tuple(tokens.shape)}")
    return tokens.reshape(t, hm, wm, -1)[0]


def pool_grid(grid, layout):
    """(h, w, d) grid -> (S, d): S=1 for 'pooled', S=4 quadrants (TL, TR, BL, BR) for 'quadrants'."""
    if layout == "pooled":
        return grid.mean(dim=(0, 1)).unsqueeze(0)
    if layout == "quadrants":
        h, w, _ = grid.shape
        hs, ws = max(h // 2, 1), max(w // 2, 1)
        cells = [grid[:hs, :ws], grid[:hs, ws:], grid[hs:, :ws], grid[hs:, ws:]]
        return torch.stack([c.mean(dim=(0, 1)) if c.numel() else grid.mean(dim=(0, 1)) for c in cells])
    raise ValueError(f"unknown target layout {layout!r}")


@torch.no_grad()
def frames_features(frames, layout, processor, vision_tower, device, compute_dtype):
    """Mean over frames of pool_grid(frame) -> (S, d)."""
    feats = [pool_grid(merged_patch_grid(f, processor, vision_tower, device, compute_dtype), layout)
             for f in frames]
    return torch.stack(feats).mean(dim=0)


def load_target_stats(path):
    if not path:
        return None
    stats = torch.load(path, map_location="cpu")
    for key in ("layout", "delta", "mean", "std"):
        if key not in stats:
            raise ValueError(f"target stats file {path} is missing {key!r}")
    return stats


@torch.no_grad()
def compute_target(example, layout, delta, stats, processor, vision_tower, device, compute_dtype):
    """(S, d) target for one example under the given layout/delta/stats."""
    if stats is not None and (stats["layout"] != layout or bool(stats["delta"]) != bool(delta)):
        raise ValueError(f"target stats were computed for layout={stats['layout']} delta={stats['delta']}, "
                         f"but this run uses layout={layout} delta={delta}")
    target = frames_features(example["future_frames"], layout, processor, vision_tower, device, compute_dtype)
    if delta:
        target = target - frames_features(example["context_frames"], layout, processor, vision_tower,
                                          device, compute_dtype)
    if stats is not None:
        target = (target - stats["mean"].to(target.device)) / stats["std"].to(target.device)
    return target


def expand_to_slots(target, n_slots):
    """(S, d) -> (n_slots, d). S=1 repeats; S=n_slots maps 1:1; otherwise slots cycle through targets."""
    s = target.shape[0]
    if s == n_slots:
        return target
    if s == 1:
        return target.expand(n_slots, -1)
    idx = torch.arange(n_slots, device=target.device) % s
    return target[idx]

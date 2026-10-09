# LatentVANS plan and experiment log

Kept up to date with every experiment: what was tried, the result, and what comes next.
Cluster rule: all work runs through Slurm (`sbatch`); nothing runs on the login node.

## Goal

Video next-event prediction (VANS setting): a VLM (Qwen2.5-VL-3B) reasons about an input video and
question, and a VDM (Wan2.1-T2V-1.3B) generates the answer video. Beat VANS (arXiv 2511.16669) on
its benchmark.

## Experiments so far

### 1. Stage 0 v1: latent grounding SFT (Sep 2026)
- Qwen2.5-VL-3B LoRA trained to predict future-frame embeddings in latent slots, on Future-L1-50K
  (run `stage0_futurel150k_run1`).
- **Result: invalid.** TwiFF frame indices were read as raw frame numbers when they are positions
  among 8 sampled frames, so context and "future" frames both came from the first ~0.3 s of each clip.
  Fixed 2026-10-02 (`scripts/twiff_frames.py`, verified on TwiFF val parquet: 92% match vs 33%).

### 2. Stage 0 v2: corrected frames (2026-10-02/03, jobs 63251/63252)
- Two targets: quadrant-pooled future embeddings (`quad_abs`) and their change from the context
  frame (`quad_delta`). 6000 steps each.
- **Result:** the latent predicts the future no better than a ridge regression on the context frame
  alone: R² 0.400 vs 0.381 (abs), 0.101 vs 0.123 (delta). Shuffling the question raises MSE by
  21–39%, so the latent uses the question but adds nothing beyond the context frame.
  Stage 0 regression tuning stopped here.

### 3. VDM conditioning pilot (2026-10-03 to 10-07)
Wan2.1-1.3B LoRA (r32), 6K Future-L1-50K clips, context frame + 16 future frames at 288x512,
3000 steps. Metric: held-out flow-matching loss on 1,323 TwiFF val clips, paired per-clip CIs.

| Conditioning | Loss change vs null | Notes |
|---|---|---|
| Stage 0 latent (abs / delta) | ≈ 0 | no information |
| caption + latent ("both") | ≈ caption | latent adds nothing |
| Qwen hidden states via projector (last / mid layer) | ≈ 0 | projector never learns past null |
| same, projector first distilled toward caption conditioning | ≈ 0 after LoRA training | beat caption at step 0 (0.3256 vs 0.3267) but fell back; real vs shuffled only 0.03–0.14% |
| caption, `future` variant | −0.45% | |
| caption (Stage 0 VLM answer) | −0.53% | beats null on 92% of clips, both seeds |
| caption, `full` variant | −0.67% | |
| caption, `oracle` variant | −0.78% | best; ≈ the most any conditioning gains here |

**Takeaway:** text helps a little but reliably; nothing taken from Qwen's internal states has helped.
Flow loss is a poor proxy for video quality, so we moved to the VANS benchmark metrics.

### 4. VANS benchmark evaluation (2026-10-08/09, jobs 66188–66246)
VANS did not release its 800-sample test split, so we rebuilt the procedural half from its released
CSVs: 200 YouCook2-validation + 200 COIN-test rows, one per video (`scripts/build_vans_eval.py`).
Metrics as in the paper (`scripts/vans_eval_metrics.py`).

| Arm | FVD ↓ | CLIP-V ↑ | CLIP-T ↑ | BLEU@4 ↑ | ROUGE-L ↑ |
|---|---|---|---|---|---|
| Paper: VANS Joint-GRPO / SFT (procedural) | 78.3 / 85.3 | 0.802 / 0.766 | 0.382 / 0.320 | 0.099 / 0.023 | 0.363 / 0.281 |
| `vans`: released model, its own captions | 1477 | 0.775 | 0.287 | 0.032 | 0.209 |
| `vans_oracle`: VANS video model, ground-truth caption | 1497 | 0.806 | 0.325 | – | – |
| `pilot_null`: our pilot VDM, no caption | 385 | 0.742 | 0.259 | – | – |
| `pilot_caption`: our pilot VDM, VANS's captions | 381 | 0.777 | 0.283 | 0.032 | 0.209 |
| `pilot_oracle`: our pilot VDM, ground-truth caption | 449 | 0.807 | 0.312 | – | – |

**Results:**
- The paper's numbers do not reproduce on our split: the released model scores below even the
  paper's SFT on ROUGE-L, CLIP-V and CLIP-T. Different test samples, and unstated details of CLIP-T
  and FVD, mean the paper's table is not a usable target as is.
- Our FVD is ~20x the paper's, and ranks our pilot far above VANS. Likely an artifact (pilot makes
  16 frames that are resampled to 33). Not trusted until videos are inspected.
- The gap is in the reasoning: ground-truth captions instead of VANS's raise CLIP-V by 0.031 and
  CLIP-T by 0.038. With the same captions, our zero-shot pilot VDM matches VANS's VDM on CLIP-V
  (0.777 vs 0.775; 0.807 vs 0.806 with ground truth).

### 5. Joint-GRPO reimplementation (code done, 2026-10-08/09)
VANS released no training code. Reimplemented from the paper on top of `code/VANS`:
`scripts/vans_sft.py` (VLM LoRA SFT, full-DiT VDM SFT), `scripts/vans_grpo.py` (stage 1: VLM GRPO
with reward format + ROUGE-L + CLIP-V of the rendered video; stage 2: Flow-GRPO on the DiT with
reward CLIP-V + CLIPScore to an anchor caption), `scripts/vans_rl_common.py`. Paper hyperparameters:
group 8, beta 0.004, clip 1e-3, lr 5e-5, LoRA r8/a32.
- All entry points pass smoke tests on 48 GB ADA6000s (jobs 66152–66187).
- Fixes needed to run the released VANS code: SDPA attention and KV cache for the VLM (eager attention
  without a cache runs out of memory), umT5 kept on CPU between encodes.
- Measured speeds: VANS generation ~95 s/video, pilot ~15 s/video, VDM SFT ~20 s/step uncached,
  GRPO stage 1 ~4 min/step at group 8 (~27 h for 800 steps on 2 GPUs).

## In progress

- Training data for SFT/RL from the COIN/YouCook2 `training` subsets (`data/vans_train`):
  2,255 / 4,000 rows; Slurm job 66505 continues it. The paper's SFT used 100K examples; its RL used 1K.

## Next steps

1. Inspect generated videos from `vans` and `pilot_*` to explain the FVD gap; fix the FVD protocol
   (frame count / resampling) so arms are comparable, then rescore all five arms.
2. Check why VANS's captions reach ROUGE-L 0.21 here vs 0.36 in the paper (decoding settings,
   prompt format, our split). Calibrate the stage-2 anchor threshold (paper: ROUGE-L ≥ 0.6, which
   would reject most prompts at the current caption quality).
3. When the training data is done: VLM SFT, then VDM SFT (with the latent/text cache), then GRPO
   stage 1, then stage 2. Evaluate after each stage with the same five-arm protocol.
4. Main comparison: our SFT → Joint-GRPO vs released VANS on the same rebuilt split, since the
   paper's absolute numbers do not carry over.
5. Since better captions are the lever, test whether our own VLM path (end-to-end training of the
   VLM against the video model's loss) improves captions beyond GRPO.

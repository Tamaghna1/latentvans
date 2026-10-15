# LatentVANS plan and experiment log

Kept up to date with every experiment: what was tried, the result, and what comes next.
Cluster rule: all work runs through Slurm (`sbatch`); nothing runs on the login node.

## Goal

Video next-event prediction (VANS setting): a VLM (Qwen2.5-VL-3B) reasons about an input video and
question, and a VDM (Wan2.1-T2V-1.3B) generates the answer video. Beat VANS (arXiv 2511.16669) on
its benchmark.

**Objective changed 2026-10-15 (user decision).** The original method (latent from the VLM into the
video model, and latent reasoning in the VLM) is dropped: four tests found no gain (experiments 2, 3,
6, 7). Caption quality is the measured lever, so the new objective is better next-event captions
through visual grounding with bounding boxes (experiment 8), scored on the VANS benchmark for both
captions and videos.

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

## Experiment 6: caption vs caption + latent on the VANS benchmark (core claim)

Both arms: DiT from released VANS, fine-tuned 4,000 steps on `data/vans_train` with identical data
order, timesteps and noise (jobs 67503 caption, 67504 latent). Scored on the 400-sample benchmark.

| Arm | Captions | FVD | CLIP-V | CLIP-T | ROUGE-L |
|---|---|---|---|---|---|
| released VANS (no fine-tune) | VANS VLM | 1477 | 0.775 | 0.287 | 0.209 |
| caption arm (fine-tuned) | VANS VLM | 544 | 0.763 | 0.277 | 0.209 |
| caption arm (fine-tuned) | ground truth | 665 | 0.777 | 0.300 | – |
| latent arm | VANS VLM | 928 | 0.760 | 0.276 | 0.209 |
| latent arm | ground truth | 1081 | 0.778 | 0.301 | – |

- The latent tokens' learned gain fell steadily during training (0.075 -> 0.068): the DiT turned the
  latent down rather than up. Early sign it adds little.
- **Result: latent adds nothing.** CLIP-V/T equal to the caption arm within noise (0.760 vs 0.763 with
  VANS captions, 0.778 vs 0.777 with GT captions). Core claim not supported in the VANS setting.
- Fine-tuning on our 4K rows slightly lowered CLIP-V/T vs the released DiT (0.763 vs 0.775).

## Experiment 7: Coconut continuous reasoning (VLM captions)

- `coconut` arm (67505, 4 stages, 2 thoughts/step): ROUGE-L **0.283**, BLEU@4 0.073 vs released VANS VLM
  0.209 / 0.032. Needs controls before reading anything into it: untrained VLM through the same eval
  path (67833: ROUGE-L 0.216, BLEU@4 0.036 -> the eval path is not the cause) and `nocot` arm trained on the same data (67834).
- **`nocot` result (68308): ROUGE-L 0.276, BLEU@4 0.070** -- almost all of Coconut's gain is fine-tuning
  on this data. Coconut - nocot = +0.007 ROUGE-L, paired bootstrap 95% CI [-0.000, +0.014]: not
  significant. **`cot` (68423): ROUGE-L 0.276, BLEU@4 0.067** -- text reasoning = no reasoning
  (cot - nocot -0.001, CI [-0.010, +0.008]); coconut - cot +0.008, CI [-0.002, +0.017]: not significant.
  Conclusion: at this scale, reasoning (text or latent) adds nothing beyond fine-tuning. `format_ok=0` only because the
  checker expects text inside [Think]; answers parse fine.

## Experiment 8: visually grounded captions with bounding boxes (in progress, from 2026-10-15)

Question: does telling the captioner which objects are where improve its next-event captions, and do
better captions give better videos? Code: `scripts/ground_objects.py`, `scripts/vans_ground.py`,
`scripts/vans_ground.sh`; W&B `latentvans-grounding`.
- Grounding pre-pass: Qwen2.5-VL-3B-Instruct (base) boxes the objects, tools, ingredients and hands in
  the last input frame at native resolution (the captioner sees 6 frames at 364x644), labels with
  color/material, boxes on a 0-1000 scale; plus a zoomed crop of the active region (hands and what
  they touch).
- Arms, all fine-tuned exactly like `nocot` (VANS VLM LoRA init, same 4,000 rows, order, 2,500 steps,
  lr 5e-5, loss on the answer):
  `ground_text` (object list with boxes added to the prompt), `ground_zoom` (+ the zoomed crop as an
  image), `ground_think` (model first writes the objects the next step will use, with boxes, then the
  caption; no detector at test time).
- Metrics: caption BLEU / ROUGE-L on the 400-sample benchmark with paired bootstrap vs `nocot`; then
  the best arm's captions rendered by the released VANS video model (CLIP-V / CLIP-T / FVD) next to
  VANS's own captions (0.775 CLIP-V), `nocot` captions and GT captions (0.806).
- Transfer check (runs in parallel): `nocot` captions rendered by the released VANS video model.

## Cluster notes

- 2026-10-10 17:03: scratch quota (~100 GB) filled while two runs saved checkpoints; the caption arm
  (66740, at step ~990), the coconut arm (66843, at step 600) and every job started afterwards died
  without logs. Freed 9 GB of smoke outputs; latent runs now keep only the newest 2.8 GB DiT
  checkpoint. Restarted as 67503 (caption), 67504 (latent), 67505 (coconut) on 2026-10-12.
  Headroom is ~12 GB: check before adding large outputs.
- cn1 (long) has a dead GPU (PCI 0000:CD:00.0, nvidia-smi "Unknown Error"); cn7 (ada) GPUs
  intermittently "busy or unavailable". Jobs exclude cn1,cn7; `scripts/gpu_probe.sh` checks a node.
- Our account cannot use h200 (needs h200_qos). Limits: 2 running / 4 submitted jobs, 2 GPUs per job.

## Possibilities (ideas to try)

Every idea raised in discussion goes here with where it came from, its status, and how we would test it.
Status: `running` / `queued` / `candidate` / `done` / `dropped`.

| # | Idea | Source | Status | How to test / notes |
|---|---|---|---|---|
| P1 | Reproduce VANS on its benchmark and set up Joint-GRPO as in the paper | user, 2026-10-08 | done (eval), code ready (RL) | Experiments 4–5. Paper numbers did not reproduce on our rebuilt split; SFT → Joint-GRPO still to run. |
| P2 | Test caption + latent vs caption in the real VANS setting (not the easier pilot) | user, 2026-10-10 | done: no gain | Experiment 6: CLIP-V 0.760 vs 0.763 (VANS captions), 0.778 vs 0.777 (GT captions). |
| P3 | Ask the VANS authors for SFT checkpoints, training code, test split | user, 2026-10-10 | candidate (email drafted) | Would let Joint-GRPO start exactly as in the paper and settle the reproduction gap. |
| P4 | Coconut continuous thoughts (arXiv 2412.06769): replace written reasoning with latent thoughts via a curriculum, no direct latent target | user, 2026-10-10 | done: no gain beyond fine-tuning; dropped 2026-10-15 | Next-steps item 0b: arms cot / nocot / coconut / pause, caption BLEU / ROUGE-L. Then (b): feed thoughts to the video model while dropping caption words. |
| P5 | ReaLVR diagnostic (arXiv 2609.34563): is the latent actually used? | user, 2026-10-10 | dropped (latent direction dropped 2026-10-15) | After P2: swap test (other sample's latent, same caption → does the video / CLIP-V change?) and sensitivity test (latent from another segment of the same video → does the latent move?). ~1 GPU-hour. |
| P6 | ReaLVR contrastive evidence loss for the latent | user, 2026-10-10 | dropped (latent direction dropped 2026-10-15) | Margin loss [m − (sim(z, f(true next clip)) − max sim(z, f(negatives)))]₊, weight ~0.2, added to the VDM loss. Hard negatives = other segments of the same video incl. the input segment, so "describe the present" cannot win (Stage 0's failure). Run if P5 shows the latent is decorative or weak. |
| P7 | If latents enter Joint-GRPO, give them a gradient path | from P6 discussion (ReaLVR's LVR baseline) | dropped (latent direction dropped 2026-10-15) | LVR's GRPO treated latents as fixed context and they went unused; use the VDM loss or an auxiliary loss during RL. |
| P8 | Use a more modern VLM (user asked about Qwen3.8) | user, 2026-10-10 | candidate (after a matched-backbone win) | Qwen3.8 smallest = 27.8B (video-capable): inference fits one A100, training needs 4-bit QLoRA + 2 GPUs. Practical upgrade: Qwen3-VL-8B / 4B (Oct 2025). For a paper, first show the method beats VANS with the same Qwen2.5-VL-3B, then show the gain holds with the bigger VLM; a win only from a bigger backbone is confounded. |
| P9 | Paper: beat VANS on its benchmark | user, 2026-10-10 | candidate | Needs (1) a gain at matched backbone, (2) ablations (P5, pause/nocot), (3) a fair benchmark: original split (P3) or our rebuilt split released with protocol, same split for VANS, (4) a literature check for novelty (latent > caption for video next-event prediction). Then scale (P8). |
| P10 | Improve captions by visual grounding with bounding boxes | user, 2026-10-15 | running | Experiment 8: arms ground_text / ground_zoom / ground_think vs nocot; best arm rendered by the VANS video model. |
| P11 | Transfer check: do better captions give better videos? | from 2026-10-15 recommendation | running | `nocot` captions (ROUGE-L 0.276 vs VANS 0.209) rendered by the released VANS video model; compare CLIP-V with VANS captions (0.775) and GT captions (0.806). |
| P12 | Detailed descriptions of what changes (objects, actions, positions) instead of a one-sentence caption, trained with video-rewarded RL | from 2026-10-15 recommendation | candidate | After P10/P11. Grounded object lists are a first version of this. |

## Next steps

1. Experiment 8: grounding pre-pass -> smoke test -> train the three grounded arms -> caption scores
   with paired bootstrap vs `nocot` -> render the best arm with the VANS video model and score.
2. P11 transfer check (parallel): if `nocot` captions raise CLIP-V toward the GT-caption ceiling,
   caption quality is confirmed as the lever for video quality too.
3. If a grounded arm wins: more training rows (we use 4,000; the paper used 100K), then Joint-GRPO
   stage 1 on the best captioner (code ready, never run in full), then a newer VLM (P8).
4. Fix the FVD protocol (frame count / resampling) so arms are comparable; check why VANS reaches
   ROUGE-L 0.21 here vs 0.36 in the paper.

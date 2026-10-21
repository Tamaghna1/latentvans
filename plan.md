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

**Objective changed 2026-10-20 (user decision; this is the user's MTech thesis, due in 1-3 months).**
Grounding gave no gain (experiment 8), and experiment 9 showed the benchmark's video metrics reward
reproducing the input scene, not showing the next event. A pure benchmark critique is not enough for
a thesis, so the thesis is analysis + evaluation + method (P26):
1. Analysis (done, experiment 9): copy, blind and fixed-caption baselines; current metrics measure
   scene match and caption boilerplate.
2. Event-centric evaluation (P27): a VLM judge scores whether the true next step happens in the
   answer video and whether the input scene is kept, validated against human labels.
3. Method: scene-anchored generation first (P28), the video model continues the input scene and
   changes only what the step changes; then an event-rewarded captioner (P29). Final system: both.
Earlier methods (latent, Coconut, grounding) become ablations / negative results.
Not tied to VANS (user, 2026-10-20: "you dont need to stick strictly to the vans paper"): any video
next-event prediction setup is fine, so other video models (e.g. image-to-video models that start
from the last frame), other datasets and other baselines are all in scope. VANS stays as a baseline
and its benchmark as one test set.

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

## Experiment 8: visually grounded captions with bounding boxes (2026-10-15/16: no gain)

Question: does telling the captioner which objects are where improve its next-event captions, and do
better captions give better videos? Code: `scripts/ground_objects.py`, `scripts/vans_ground.py`,
`scripts/vans_ground.sh`; W&B `latentvans-grounding`.
- Grounding pre-pass (job 68487, done 2026-10-15): on the last input frame, Qwen2.5-VL-3B-Instruct
  (base) names the objects, tools, food and hands involved (given the user's instruction), then
  Grounding DINO (base) boxes each name; names without a box scoring >= 0.3 are dropped, which also
  filters names the VLM invented. Boxes on a 0-1000 scale, plus a zoomed crop of the active region
  (hands and what they touch). 4.8 objects/frame; ground_think targets keep 2.0 next-step objects
  per row (17% of rows have none visible). Note: most YouCook clips are 640x360, so the last frame
  adds little resolution over the captioner's 364x644 video frames; the crop adds tokens on the
  active region.
  - Dead ends on the way: asked to box everything itself, Qwen2.5-VL-3B returned generic labels
    ("object", "tool", "food") with every prompt tried (jobs 68481, 68483), so detection was moved
    to Grounding DINO. First attempts also failed on a missing ffmpeg in the job PATH and an OOM in
    the vision tower's batch-wide attention mask on 24 GB GPUs (batch 12 -> 4).
- Smoke test passed (68491): grounded prompt reproduces VANS's tokenization exactly without the
  extras; the zoom crop adds 242 image tokens. Training: 68500 (ground_text), 68501 (ground_zoom),
  68502 (ground_think), all done 2026-10-16. Caption evals: 68651 (ground_text), 68626 (ground_zoom),
  68637 (ground_think), all done. ground_zoom's captions rendered by the VANS video model: 68660 (done).
- **Caption results (400 samples, paired bootstrap vs `nocot`, 95% CI):**

  | Arm | ROUGE-L | BLEU@4 | ROUGE-L - nocot | Train loss (last 500 steps) |
  |---|---|---|---|---|
  | `nocot` | 0.276 | 0.070 | – | 1.042 |
  | `ground_text` | 0.277 | 0.075 | +0.001 [-0.006, +0.008] | 1.042 |
  | `ground_zoom` | 0.282 | 0.071 | +0.006 [-0.002, +0.014] | 1.037 |
  | `ground_think` | 0.270 | 0.068 | -0.006 [-0.015, +0.002] | (incl. object list) |
  | `coconut` (exp. 7, for scale) | 0.283 | 0.073 | +0.006 [-0.001, +0.014] | – |

  No significant gain. The object list does not even lower the training loss (1.042 = nocot), so the
  model gets nothing from it that it does not already see in the video. ground_zoom's +0.006 is not
  from grounding: it is no larger on the 306 rows where a next-step object was detected in the last
  frame (+0.007) than on the 94 where none was (+0.004), and coconut (no grounding) shows the same
  +0.006. (Not seed spread: two `nocot` seeds differ by only 0.001, P24; the +0.006 is inside the
  ±0.008 test-sample noise of 400 samples, so it is n.s., possibly a tiny shared fine-tuning effect.)
  Reading: the captioner already
  perceives the scene; what it gets wrong is what happens next, which object boxes do not supply.
  ground_think (predict the next step's objects first) is slightly worse, significantly on YouCook
  (-0.013 [-0.024, -0.002]). **Conclusion: grounding with boxes does not improve next-event captions;
  grounding stopped.** The ground_zoom render (68660) completes the benchmark row: see the transfer table.
- Arms, all fine-tuned exactly like `nocot` (VANS VLM LoRA init, same 4,000 rows, order, 2,500 steps,
  lr 5e-5, loss on the answer):
  `ground_text` (object list with boxes added to the prompt), `ground_zoom` (+ the zoomed crop as an
  image), `ground_think` (model first writes the objects the next step will use, with boxes, then the
  caption; no detector at test time).
- Metrics: caption BLEU / ROUGE-L on the 400-sample benchmark with paired bootstrap vs `nocot`; then
  the best arm's captions rendered by the released VANS video model (CLIP-V / CLIP-T / FVD) next to
  VANS's own captions (0.775 CLIP-V), `nocot` captions and GT captions (0.806).
- Transfer check (job 68479, done 2026-10-16): `nocot` captions rendered by the released VANS video
  model (`results/vans_eval/captions_nocot`). Paired bootstrap over the 400 samples, 95% CI:

  | Captions into the VANS video model | ROUGE-L | CLIP-V | CLIP-T | FVD |
  |---|---|---|---|---|
  | VANS VLM (released) | 0.209 | 0.775 | 0.287 | 1477 |
  | `nocot` (our fine-tuned VLM) | 0.276 | 0.781 | 0.292 | 1434 |
  | `ground_zoom` (68660) | 0.282 | 0.785 | 0.293 | 1298 |
  | ground truth | – | 0.806 | 0.325 | 1497 |

  nocot - VANS: CLIP-V +0.006 [+0.001, +0.011], CLIP-T +0.005 [+0.002, +0.008]; GT - nocot: CLIP-V
  +0.025 [+0.020, +0.030]. **Better captions do give better videos, but weakly:** +0.067 ROUGE-L
  closes ~19% of the CLIP-V gap to ground-truth captions. Caption quality is the right lever, and
  ROUGE-L gains have to be large to move video metrics. ground_zoom - nocot: CLIP-V +0.0045
  [+0.0003, +0.0085] (borderline, and its ROUGE-L gain was n.s.), CLIP-T +0.002 [-0.001, +0.004];
  the frozen frame (Experiment 9) still beats it on CLIP-V by 0.025 [0.018, 0.032].

## Experiment 9: does the benchmark measure next-event prediction? (from 2026-10-16)

Diagnostics for the paper framing (P16, P23, P24, P25). Code: `scripts/copy_baselines.py/.sh`, blind arm
in `scripts/vans_ground.py`.
- **Copy baselines (P16, job 68664):** answers that predict nothing, scored with the same metrics.
  Paired over the same 400 samples (95% CI):

  | Answer video | CLIP-V | CLIP-T | FVD |
  |---|---|---|---|
  | VANS video model, VANS captions | 0.775 | 0.287 | 1477 |
  | VANS video model, `nocot` captions | 0.781 | 0.292 | 1434 |
  | VANS video model, ground-truth captions | 0.806 | 0.325 | 1497 |
  | `copy_last`: last input frame, frozen for 33 frames | **0.811** | 0.289 | 903 |
  | `copy_input`: the input clip itself | 0.810 | 0.287 | **237** |

  copy_last - VANS: CLIP-V +0.035 [+0.028, +0.043] (higher on 73% of samples); copy_last - GT-caption
  videos: +0.004 [-0.003, +0.012]. CLIP-T: copy_last - VANS +0.002 [-0.002, +0.006], copy_last - nocot
  -0.003 [-0.006, +0.001]; only GT-caption videos are higher (+0.036 [+0.032, +0.040]).
  **A frozen frame beats every model on CLIP-V and ties ground-truth-caption videos; CLIP-T cannot
  tell the models' videos from no prediction; the input clip has the best FVD.** On this split the
  video metrics reward reproducing the scene, not showing the next event. CLIP-T separates only
  videos made from the right caption, so it is the one video metric with some signal. Caveats: our
  rebuilt split, not the paper's (P3); real frames also look sharper than generated ones, which CLIP-V
  may reward (ruled out by the 68809 control below). Next: does the same hold on the paper's split / other video-as-answer benchmarks, and
  which metric does track the event (VLM judge, P25)?
- **Text-only baselines (P23, `scripts/text_baselines.py`, no model, no video):** ROUGE-L on the same
  400 samples against VANS's ground-truth captions (VANS's own `ENG_GT_Caption`, so the caption style is
  theirs, not ours):

  | Captions | ROUGE-L | vs released VANS (0.209), paired 95% CI |
  |---|---|---|
  | released VANS VLM | 0.209 | - |
  | `const`: one fixed training caption for every sample ("a person is using a spoon and fork to place carrots and zucchini into a white bowl ...") | 0.231 | +0.022 [+0.014, +0.030] |
  | `nn_target`: training caption retrieved by the true next-step sentence (oracle step, no video) | 0.233 | +0.025 [+0.015, +0.035] |
  | `nn_instr`: training caption retrieved by the instruction text | 0.211 | - |
  | `task_central`: most central training caption (max mean ROUGE-L to the others) of the same task, e.g. all "corn dogs" rows (no model, no video) | 0.243 | +0.034 [+0.026, +0.043] |
  | `target_sent`: the true next-step sentence itself ("deep fry the corn dog in oil") | 0.157 | - |
  | `nocot` (our fine-tuned VLM) | 0.276 | +0.067 [+0.057, +0.077] |

  **One fixed, wrong caption beats the released VANS model**, and the correct step written plainly
  scores far below it: ROUGE-L mostly rewards the dataset's caption style and scene boilerplate. E.g.
  on the corn-dog sample VANS writes "A black electric deep fryer containing hot oil ..." (the right
  event) and still loses to the carrots caption. Knowing the true step (nn_target) adds only +0.002 over
  the constant. Fine-tuned captioners sit 0.045 above the constant; that is the whole room in which
  every captioner variant we tried (cot, coconut, grounding) moved by <= 0.007. Train/test share 2 of
  400 videos and all 203 test tasks appear in training.
- **Blind captioner (P23, 68665 train, 68680 eval):** `nocot` trained and evaluated with the video
  removed (instruction text only), same data and steps. ROUGE-L **0.246**, BLEU@4 0.048; vs `nocot`
  -0.030 [-0.039, -0.021] (YouCook -0.045, COIN -0.015); vs the fixed caption +0.015 [+0.006, +0.024];
  vs released VANS +0.037. Training loss 1.141 vs `nocot` 1.042. Ladder: fixed caption 0.231 -> text
  only 0.246 -> with video 0.276. **The video is worth ~0.03 ROUGE-L, 4x more than any captioner
  variant we tried (<= 0.007), but small next to the 0.231 a fixed caption gets.** So the caption
  metric carries some visual signal, unlike CLIP-V; its floor is just very high.
- **Fixed-caption video (P16 + P23, 68768):** the `const` caption rendered by the released VANS video
  model for every sample (same input clips). If its CLIP-V/CLIP-T land near the models' videos, the
  video metrics do not depend on what the caption predicts. Output `results/vans_eval/captions_const`.
  **Done (2026-10-17 01:09):** CLIP-V 0.688, CLIP-T 0.230, FVD 5271 (VANS 0.775 / 0.287 / 1477). Paired
  vs VANS: CLIP-V -0.087 [-0.097, -0.078] (COIN -0.144, YouCook -0.031), CLIP-T -0.057 [-0.062, -0.052];
  below VANS on 344/400 samples. **The video metrics do depend on the caption, through the scene:** a
  cooking caption on COIN's non-cooking tasks is punished hard, and on YouCook (still kitchens, wrong
  dish) far less. With the copy baselines: right scene and no event (frozen frame) >= every model,
  wrong scene << every model, so CLIP-V/CLIP-T measure scene match, not whether the next event
  happens. **The two halves of the benchmark also disagree:** ROUGE-L ranks the fixed caption above
  released VANS (0.231 vs 0.209), the video metrics rank its video far below (0.688 vs 0.775).
- **Sharpness / compression control for the copy baselines (68809, `scripts/vae_copy_baselines.py`):** does
  the frozen frame win CLIP-V only because real frames are sharper? Arms: `copy_last_q5` (saved like
  generated videos: 11 fps, quality 5), `copy_last_vae` (+ Wan2.1 VAE encode/decode, what a model that
  copied the frame perfectly would output), `copy_input_vae`. Smoke (68808) ok; by eye the VAE round
  trip barely softens the frame. Also seen by eye on youcook_xkKuIlYSMMU_1: the VANS video model draws a
  new scene (corn dog on a blue-rimmed plate) rather than continuing the input's (glass bowl on a
  wooden board), which fits the copy baselines winning CLIP-V.
  **Done (2026-10-16 21:57):** paired over the 400 samples (95% CI):

  | Answer video | CLIP-V | CLIP-T | FVD | CLIP-V vs VANS | CLIP-V vs GT-caption videos |
  |---|---|---|---|---|---|
  | `copy_last` (near-lossless) | 0.811 | 0.289 | 903 | +0.035 [+0.028, +0.043] | +0.004 [-0.003, +0.012] |
  | `copy_last_q5` | 0.806 | 0.288 | 904 | +0.031 [+0.024, +0.038] | -0.000 [-0.008, +0.007] |
  | `copy_last_vae` | 0.809 | 0.288 | 912 | +0.034 [+0.026, +0.041] | +0.002 [-0.005, +0.009] |
  | `copy_input_vae` | 0.806 | 0.288 | 240 | +0.031 [+0.024, +0.037] | -0.001 [-0.007, +0.006] |

  VAE + compression cost the frozen frame 0.002 CLIP-V (q5 alone 0.004; VAE-smoothed frames compress
  better). `copy_last_vae` still beats VANS on 285/400 samples, `nocot`-caption videos by +0.028
  [+0.021, +0.035] and `ground_zoom`-caption videos by +0.023 [+0.017, +0.030]; on COIN it beats even
  GT-caption videos (+0.011 [+0.002, +0.020]; YouCook -0.006 [-0.017, +0.005]). CLIP-T: vs VANS +0.001
  [-0.003, +0.004], vs `nocot` -0.004 [-0.007, -0.000]. **Sharpness is not why the copies win: a video
  model that output the last input frame, unchanged, would top the CLIP-V table and tie every model on
  CLIP-T.**
- **Event judge (P27, 69637, Qwen2.5-VL-3B, done 2026-10-21 03:28):** P(yes) to "does this video show
  <next-step sentence>" (p_next), the same for the current step (p_current), and "same scene as the last input
  frame" (p_scene); 8 frames per video. Means over the 400 samples (yes = share with p_next > 0.5):

  | Answer video | p_next | yes | p_current | p_scene | CLIP-V |
  |---|---|---|---|---|---|
  | real next clip (target) | **0.704** | 0.840 | 0.478 | 0.492 | - |
  | VANS video model, GT captions | 0.659 | 0.765 | 0.418 | 0.520 | 0.806 |
  | input clip itself (`copy_input`) | 0.397 | 0.412 | **0.739** | 0.525 | 0.810 |
  | `nocot` captions -> VANS video model | 0.388 | 0.410 | 0.422 | 0.531 | 0.781 |
  | released VANS | 0.353 | 0.360 | 0.471 | 0.525 | 0.775 |
  | frozen last frame (`copy_last`) | 0.324 | 0.282 | 0.537 | 0.544 | 0.811 |
  | one fixed caption for all (`const`) | 0.079 | 0.030 | 0.080 | 0.408 | 0.688 |

  Paired p_next: target - copy_last +0.380 [+0.349, +0.411]; GT-caption videos - copy_last +0.334 [+0.302,
  +0.368]; target - GT-caption videos +0.045 [+0.021, +0.070]; GT-caption videos - `nocot` +0.271 [+0.235,
  +0.306]; `nocot` - VANS +0.035 [+0.001, +0.069]; VANS - copy_last +0.029 [-0.001, +0.060] (n.s.); `nocot` -
  copy_input -0.009 n.s. On p_next - p_current, VANS beats copy_last by +0.095 [+0.056, +0.133] and `nocot`
  beats copy_input by +0.308 [+0.265, +0.349]. **p_next ranks what CLIP-V cannot: the real next clip and
  right-caption videos far above the copies, the fixed caption at the bottom, and p_current is highest for the
  input clip (the current step), as it should be. Released VANS shows the next step barely more often than the
  frozen frame (n.s.) and no more than replaying the input; the caption is the main lever (`nocot` -> GT
  captions +0.27, same video model).** (Correction: the partial run's 0.256 for VANS came from its first 178
  rows; on all 400 it is 0.353.) **p_scene is useless with the 3B judge** (0.49-0.54 for everything, incl. the
  frozen frame, which is the input frame): replace it with an embedding similarity to the last input frame.
  Not yet validated against human labels.
- **Scene retention (`scene_sim.py`, 69685/69686, done 2026-10-21):** DINOv2-S cosine between the last input frame
  and the answer frames (CLIP image cosine in brackets), 400 samples: real next clip 0.517 (0.815), GT-caption
  videos 0.489 (0.798), `nocot` 0.510 (0.804), VANS 0.506 (0.800), input clip 0.688 (0.874), frozen frame 0.992
  (0.987), VAE frozen frame 0.986, fixed caption 0.215 (0.697). VANS - real next clip -0.011 [-0.034, +0.012]
  (n.s.). **On average VANS keeps as much of the input scene as the real next clip does; the real next step
  often changes the view or place (DINO 0.27 in the lowest third of samples, 0.76 in the highest).** What VANS
  misses is *which* samples stay in scene: Spearman(real clip's scene score, generated video's) is +0.39 for
  VANS, +0.43 for `nocot`, +0.70 for GT-caption videos. By thirds of the real clip's scene score (low / mid /
  high): real 0.27 / 0.52 / 0.76, VANS 0.40 / 0.53 / 0.58, GT captions 0.30 / 0.52 / 0.64. The frozen frame's
  CLIP-V comes from the samples whose next step stays in place (0.743 / 0.806 / 0.883 by the same thirds; VANS
  0.732 / 0.776 / 0.818). The right caption also carries whether the scene changes.
- **Second `nocot` seed (P24, 68707 train, 68708 eval, done):** ROUGE-L 0.277 (BLEU@4 0.068) vs seed 0
  0.276: +0.001 [-0.007, +0.009]. Training seed spread is small; the ±0.008 CIs come from the 400-sample
  test set. Every captioner variant vs seed 1: coconut +0.005 [-0.003, +0.014], ground_zoom +0.005
  [-0.003, +0.013], ground_text +0.000, cot -0.002, ground_think -0.007 [-0.016, +0.002]: none
  significant against either seed. Detecting a +0.005 effect would need roughly 4x the test samples.

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
| P10 | Improve captions by visual grounding with bounding boxes | user, 2026-10-15 | done: no gain | Experiment 8: ground_text +0.001, ground_zoom +0.006, ground_think -0.006 ROUGE-L vs nocot, all n.s.; ground_zoom render (68660): CLIP-V 0.785 vs nocot 0.781, +0.0045 [+0.0003, +0.0085], CLIP-T n.s. |
| P11 | Transfer check: do better captions give better videos? | from 2026-10-15 recommendation | done: yes, weakly | `nocot` captions (ROUGE-L 0.276 vs VANS 0.209) in the released VANS video model: CLIP-V 0.781 vs 0.775 (CI [+0.001, +0.011]), GT captions 0.806. ~19% of the gap to GT closed. |
| P12 | Detailed descriptions of what changes (objects, actions, positions) instead of a one-sentence caption, trained with video-rewarded RL | from 2026-10-15 recommendation | candidate | After P10/P11. Grounded object lists are a first version of this. |
| P13 | GRPO stage 1 on the best captioner, text rewards first (ROUGE-L + format, no rendered video), then add the CLIP-V reward | 2026-10-16 literature review (VANS Table 2a; SCST-style metric RL) | candidate, top priority | VANS: SFT -> stage 1 = ROUGE-L +0.082, BLEU@4 +0.075, CLIP-V +0.015, CLIP-T +0.032 (VDM frozen, so the video gain comes from captions alone). Our `nocot` (0.276) ≈ their SFT (0.281). Text-only rewards need no video rendering, so it is far cheaper than the full stage 1 (~4 min/step). Watch for templated captions (keep KL, check CLIP-V/T transfer). |
| P14 | GRPO stage 2 (Flow-GRPO on the DiT): reward CLIP-V to the target video + CLIPScore to the anchor caption | VANS Table 2a | candidate, after P13 | VANS: CLIP-V +0.022, CLIP-T +0.030, FVD -1.9 over stage 1. Most expensive step (video rollouts, group 8, ~1K steps). Needs the FVD protocol fixed first to report FVD. |
| P15 | Test-time selection: MBR captions (N samples, pick the one with the highest mean ROUGE-L to the others); best-of-N videos picked by the medoid (CLIP-V to the other samples) or CLIPScore to the caption | MBR decoding literature (arXiv 2311.05263, 2401.02749) | candidate, cheap for captions | No training. Captions: N=8–16 samples of a 3B VLM, minutes. Videos cost N x ~95 s per sample. |
| P16 | Diagnostic: copy-the-last-input-frame baseline (static video) through CLIP-V / FVD | 2026-10-16 review | done: frozen frame beats all models | Experiment 9: copy_last CLIP-V 0.811 vs VANS 0.775, nocot captions 0.781, GT captions 0.806; CLIP-T ties the models (0.289 vs 0.287/0.292); input clip FVD 237 vs VANS 1477. Survives the VAE + compression control (68809): copy_last_vae 0.809, +0.034 [+0.026, +0.041] over VANS. |
| P17 | More SFT rows: 4K -> the full procedural training set (VANS used 30K procedural of 100K) | VANS setup | candidate | Run before P13 if cheap enough; the RL starting point matters. |
| P18 | Stage 1 reward fixes: (a) drop the format reward or make it a gate (`nocot` never writes [Think], so r_format is 0 except on rare samples, where its +1 swamps ROUGE differences of ~0.05); (b) render all G captions of a group with the same seed (now seed + k, so CLIP-V differences partly measure noise); (c) normalize each reward within the group before summing (GDPO, arXiv 2601.05242), else ROUGE's larger spread drowns CLIP-V; (d) add BLEU@4 (a benchmark metric) to the reward; (e) several prompts per step instead of 1 | 2026-10-16 objective review (user asked to research the RL objective) | candidate, do with P13 | Log each reward's within-group std first to confirm (c). (a) and (b) are bugs for our setup and are cheap to fix. |
| P19 | Visual reward without rendering: CLIP similarity between the caption and the ground-truth target frames | 2026-10-16 objective review | candidate | Rewards captions that describe what the target video shows, at text-reward cost; can replace or thin out the rendered CLIP-V reward. Compare against rendered CLIP-V on a few hundred captions first. |
| P20 | Stage 2 on the captions the video model sees at test time: condition on VLM-sampled captions (no ROUGE-L >= 0.6 anchor filter), reward CLIP-V to the GT video + CLIPScore to the GT caption | 2026-10-16 objective review | candidate | VANS trains stage 2 only on near-perfect anchor captions; our captions average ROUGE-L 0.28, so few pass 0.6 and test-time captions are unlike training ones. Teaches the video model to fix caption errors from the input frames. Target: part of the 0.025 CLIP-V gap between our captions and GT captions. |
| P21 | Stage 2 optimizer: reward gradients (VADER, arXiv 2407.08737; CLIP-V/T are differentiable) or DiffusionNFT (arXiv 2509.16117, 3–25x more efficient than Flow-GRPO) / MixGRPO (arXiv 2507.21802) instead of Flow-GRPO | 2026-10-16 objective review | candidate | Flow-GRPO with group 8 is the most expensive part of the plan. VADER: 1 sample per prompt, truncated backprop, LoRA. Pick one after P20 shows stage 2 is worth it. |
| P22 | Reward-hacking guard: train with a different CLIP (ViT-L/14 or SigLIP, + DINOv2) than the ViT-B/32 used for scoring; report both | 2026-10-16 objective review (Pref-GRPO arXiv 2508.20751 on hacking) | candidate | A gain that shows only under the training CLIP is reward hacking, not better videos. |
| P23 | Blind baseline: predict the next-step caption from the instruction text alone (no video), and a "repeat the current step" caption | 2026-10-16 paper-direction review (user asked "am i on the right track") | done: fixed caption 0.231 (beats released VANS 0.209), blind VLM 0.246, nocot 0.276; video worth ~0.03 ROUGE-L | If text-only gets close to `nocot`'s 0.276 ROUGE-L, the caption metric mostly measures language priors, not watching the video. Decides the paper framing together with P16 and P24. |
| P24 | Noise floor: retrain `nocot` with a second seed (data order) and score it | 2026-10-16 paper-direction review | done: seeds differ by 0.001 ROUGE-L [-0.007, +0.009]; no captioner variant beats either seed | Every captioner change so far (cot, coconut, ground_text, ground_zoom) lands within ±0.006 ROUGE-L of `nocot`; a second seed shows whether that is just run-to-run spread. Any method claim needs gains clearly above it. |
| P25 | Paper framing B: "video-as-answer benchmarks do not measure next-event prediction", with an event-centric evaluation (VLM judge: did the predicted step happen; object-state checks) and baselines across VANS and other models | 2026-10-16 paper-direction review | folded into P26 (2026-10-20): analysis + motivation chapter, not the whole thesis | Pursue if copy-frame (P16) and blind (P23) baselines score close to the models. Our negative results (reasoning, latent, grounding add nothing; caption gains transfer weakly to video) become evidence. Otherwise framing A: method paper via P13/P19/P20 on a fair split (P3). Scope: every diagnostic so far is on the procedural half (COIN/YouCook2); VANS's predictive half was not released, so a general claim needs predictive data (ask the authors or rebuild). |
| P26 | Thesis objective: analysis (experiment 9) + event-centric evaluation (P27) + method (P28 then P29); "it should not be just benchmarking" | user, 2026-10-20 (MTech thesis, due in 1-3 months) | running | Main claim to aim for: the first model that beats the frozen-frame baseline on CLIP-V and also shows the right next event (P27 judge). Earlier negative results are ablations. |
| P27 | Event-centric evaluation (`scripts/event_judge.py/.sh`; judge = Qwen2.5-VL-3B-Instruct, user decision 2026-10-20, no 7B; smoke 69634 ok, all arms 69637 done 2026-10-21, see experiment 9): VLM judge (P(yes)) asks (a) does the answer video show the true next step (benchmark's step sentence), (b) is it the same scene as the input's last frame | 2026-10-20 thesis plan | running: (a) passes the ordering check (target > GT-caption videos > model videos > copies > fixed caption), (b) fails with 3B (all ~0.5) | Score every existing arm (copy_last, copy_input, const, vans, nocot, GT-caption videos). Must rank GT-caption videos > model videos > copies on (a), and copies high on (b). Validate on ~100 samples labelled by the user (agreement / correlation). Label set (2026-10-21, `scripts/make_label_set.py`, seed 0): 120 distinct samples, 20 per arm (target_real, vans_oracle, vans, captions_nocot, copy_input, copy_last; 10 COIN + 10 YouCook each), shuffled, arm hidden; same question as p_next; videos cropped to 640x352 like the judge's input; the 20 copy_last items are still images (checked: exactly those 20 have no motion), and the page tells the labeller to answer Yes for a still only if the image itself shows the action being done. Mapping in `results/vans_eval/human_labels/items.json`; labels on a private claude.ai page (https://claude.ai/artifact/LX7BEh8b6d3SnNev8sdejn), read back by Claude. Final (2026-10-21, all 120 labelled: 45 yes, 54 no, 21 unsure; `scripts/judge_agreement.py`, unsure left out, n=99): AUC of p_next vs human 0.794 [0.697, 0.877], accuracy at 0.5 0.697 [0.606, 0.788], Cohen's kappa 0.396 [0.217, 0.564] (fair). Counting unsure as no: AUC 0.786, kappa 0.35; as yes: AUC 0.711, kappa 0.30 (unsure items are mostly frozen frames (8) and GT-caption videos (6)). Human yes rate by arm: real clip 0.90, GT-caption videos 0.64, input clip 0.33, `nocot` 0.31, frozen frame 0.25, VANS 0.21; judge yes rate: 0.90, 0.86, 0.33, 0.44, 0.17, 0.32. The judge ranks real clips high and copies/VANS low as people do, but **over-credits videos rendered from the right caption (0.86 vs 0.64, agreement only 0.50 on that arm)**: it partly rewards the step's objects being drawn rather than the action being done. A judge reward on rendered videos could be hacked the same way (P29 uses a text reward). Labels saved as `results/vans_eval/human_labels/labels.json`, report `agreement.json`. (b) is replaced by `scripts/scene_sim.py`: DINOv2-S and CLIP cosine between the last input frame and the 33 answer frames (mean, and last 8 frames for drift), as in VBench's consistency dimensions but against the input; the real next clip gives the natural level (69685 all arms, 69686 new anchoring arms). Eval judge differs from any training reward judge (P29). |
| P28 | Scene-anchored generation: the video model continues the input scene instead of redrawing it | 2026-10-20 thesis plan; seen on youcook_xkKuIlYSMMU_1 (VANS draws a new plate and table) | running: training-free step 1 done with GT captions (working strength 0.85) | Step 0 (done 2026-10-20): VANS encodes the input clip (33 frames -> 9 latents), prepends 6 of them, clean, to the noisy latents at every step at the same timestep, and drops them after; nothing ties the output's first frame to the input, so it can redraw the scene. Step 1 (no training, `scripts/anchor_generate.py/.sh`): (a0) first-frame replacement: reset the first generated latent to the last input frame's latent, noised to each step's level; (a) start denoising from the noised VAE latent of the frozen last frame (SDEdit-style, strength sweep), captions from `nocot`; (b) an off-the-shelf image-to-video model started from the last input frame (e.g. Wan2.2-TI2V-5B, Wan2.1-I2V, CogVideoX-I2V; check licence, size vs our 24-48 GB GPUs) with the same captions. Not tied to VANS's video model (user, 2026-10-20). Step 2: fine-tune the DiT (LoRA) with the last input frame as a clean first latent frame, on data/vans_train target clips. Win = CLIP-V above copy_last (0.811) and higher P27 event score than VANS. Smoke (69635, ffrep + SDEdit 0.5, `nocot` captions, 2 YouCook samples, by eye): the video stays in the input scene (same bowl / pan / stove) where VANS and plain `nocot` redraw it; on the corn-dog sample the step needs a new place (the fryer) and the anchored video morphs the batter in the bowl instead: anchoring can block steps that change location. Screen on 100 samples (every 4th row; 50 COIN + 50 YouCook): ffrep (69638, done) CLIP-V 0.784 vs `nocot` 0.779 (+0.005 [+0.002, +0.009]), VANS 0.777, copy_last 0.816 (-0.032 [-0.046, -0.019]); CLIP-T +0.001 n.s.; judge p_next 0.425 vs `nocot` 0.371 (+0.053 [+0.028, +0.079]) and copy_last 0.354 (+0.070 [+0.012, +0.131]). Pinning the first frame anchors only that frame; the rest still drifts. SDEdit 0.5 (69639, done): CLIP-V 0.814 vs `nocot` +0.035 [+0.023, +0.048], vs copy_last -0.002 [-0.007, +0.002] (tie); CLIP-T +0.000 n.s.; p_next 0.392 vs `nocot` +0.020 [-0.032, +0.074] n.s., vs copy_last +0.037 [+0.001, +0.075]; but p_current 0.540, near copy_last's 0.547 (`nocot` 0.455): at 0.5 the video stays close to the frozen frame. First training-free arm to tie the frozen frame on CLIP-V while showing the next step more often. Subset references: GT-caption videos p_next 0.696 / CLIP-V 0.800, target 0.732. SDEdit 0.7, `nocot` (69684, done): CLIP-V 0.804 (+0.026 [+0.015, +0.037] vs `nocot`, -0.012 [-0.020, -0.004] vs copy_last), p_next 0.425 (+0.053 [+0.005, +0.101] vs `nocot`, +0.033 [+0.002, +0.064] vs SDEdit 0.5), DINO scene 0.756: best training-free variant so far. SDEdit 0.5, GT captions (69683, done): p_next 0.482 vs GT captions without anchoring 0.696 (-0.214 [-0.284, -0.145]), CLIP-V 0.815 (+0.014 vs GT captions, ties copy_last), CLIP-T -0.026: **with the right caption, strong anchoring removes a third of the event signal**. Anchored videos sit far above the real clip's scene level (DINO 0.76-0.83 vs 0.52): uniform anchoring overshoots. Oracle adaptive anchoring (P30) gains nothing over uniform 0.7. Strength sweep with GT captions (69683, 69697, 69698, done 2026-10-21; same 100 samples): p_next 0.482 / 0.565 / 0.666 at strength 0.5 / 0.7 / 0.85 vs 0.696 without anchoring; CLIP-V 0.815 / 0.812 / 0.810 vs 0.800; DINO scene 0.818 / 0.746 / 0.660 vs 0.494 (real clip 0.523). **SDEdit 0.85 + GT captions keeps the event (p_next -0.030 [-0.071, +0.010] vs GT captions unanchored, n.s.) and ties the frozen frame on CLIP-V (-0.006 [-0.019, +0.005]) while beating it on p_next by +0.312 [+0.254, +0.372]**: with good captions, light anchoring gives both. Each step 0.85 -> 0.7 -> 0.5 costs ~0.08-0.10 p_next (both steps significant) for only +0.002-0.003 CLIP-V (n.s.). Anchoring still overshoots scene retention even at 0.85 (mean |DINO(gen) - DINO(real)| per sample 0.19 vs 0.13 unanchored). With the same strength 0.7, GT captions beat `nocot` by +0.140 p_next [+0.090, +0.191]: the caption is still the bigger lever. Judge caveat (P27): it over-credits GT-caption videos, but both arms of the 0.85-vs-unanchored comparison use GT captions. Next (69748, submitted 2026-10-21): SDEdit 0.85 with `nocot` captions, to complete the grid; working choice for the final system: strength 0.85. |
| P29 | Event-rewarded captioner: GRPO on the captioner with a judge reward (does the caption describe the true next step / does its rendered video show it) instead of ROUGE-L | 2026-10-20 thesis plan; builds on P13/P18/P19 | candidate, after P28 (may move first: the caption is the bigger lever, experiment 9 judge) | Training judge differs from the P27 eval judge (reward hacking, cf. P22). Compare to `nocot` on P27, ROUGE-L, and through the P28 video model. Reward check (2026-10-21, 800 rendered captions: `nocot` + VANS, no training): Spearman with the judge's p_next of the rendered video: ROUGE-L to the GT caption (VANS's stage-1 reward) +0.174 [+0.106, +0.247]; ROUGE-L to the step sentence +0.158; share of the step sentence's content words found in the caption +0.366 [+0.305, +0.430] (`nocot` +0.427, VANS +0.292); same share for the current step +0.183. **ROUGE-L, the reward VANS optimizes, barely tracks whether the video shows the next step; naming the next step's action and objects tracks it twice as well.** Word overlap is hackable (word lists), so the reward should be a text model's judgment of "does the caption describe <step>" (a different model from the eval judge) with KL; word recall as a cheap baseline reward. Training rows carry the step sentence too (build_vans_eval.py writes target_sentence for both splits). |
| P30 | Adaptive anchoring: anchor only where the next step stays in the input's place (strength per sample) | 2026-10-21, from the scene-retention result and the corn-dog smoke sample | checked with an oracle, no gain | Oracle on the 100-sample subset (anchored video where the real clip's DINO scene score > thr, else `nocot`): SDEdit 0.7, thr 0.5: CLIP-V 0.799, p_next 0.423, both below uniform SDEdit 0.7 (0.804, 0.425); thr 0.6/0.7 lower still; SDEdit 0.5 mixes 0.794-0.806 / 0.389-0.402. With `nocot` captions, anchoring does not cost the event on scene-change samples, so choosing per sample buys nothing; revisit with better captions (GT-caption anchoring loses 0.21 p_next at 0.5). |

## Next steps

Thesis plan (2026-10-20, P26), roughly by week:
1. Week 1: P28 step 0-1 (training-free anchoring, strength sweep) and P27 judge on all existing arms;
   the user labels ~100 samples to validate the judge.
2. Weeks 2-4: P28 step 2 (anchored DiT fine-tune); score with CLIP-V/CLIP-T/FVD and P27.
3. Weeks 4-7: P29 event-rewarded captioner; combine with P28 for the final system.
4. Weeks 7+: ablations (earlier methods), FVD protocol fix, writing. Predictive data beyond the
   procedural half only if time allows.

Earlier next steps (kept for the record):

1. Experiment 8 (grounding): done, no gain. ground_zoom render (68660, done): CLIP-V 0.785 vs nocot
   0.781 (+0.0045, borderline), CLIP-T n.s.
2. Decide the paper framing (P25) with three diagnostics, running since 2026-10-16:
   copy-the-frame video baselines (P16: done, a frozen frame beats every model on CLIP-V and ties them
   on CLIP-T, and survives the VAE/compression control; a fixed wrong caption's video scores far below
   the models -> the video metrics measure scene match, not the event; leaning to the critique framing), blind no-video captioner (P23, done: video worth ~0.03 ROUGE-L; a fixed caption beats released VANS),
   second `nocot` seed for the noise floor (P24, done: seeds differ by 0.001; test-set noise ±0.008).
   - Copy / blind baselines close to the models -> benchmark-critique paper with an event-centric
     evaluation (P25); our negative results (experiments 6-8) are its evidence.
   - Baselines far below -> method paper: caption RL with text rewards first (P13, P18, P19), then
     stage 2 on test-time captions (P20), on a fair split (P3); gains must clear the P24 noise floor.
3. Fix the FVD protocol (frame count / resampling) so arms are comparable; check why VANS reaches
   ROUGE-L 0.21 here vs 0.36 in the paper.

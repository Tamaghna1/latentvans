"""Scene-copy baselines for the VANS benchmark's video metrics (plan.md P16): how much CLIP-V / CLIP-T / FVD
an answer gets without predicting anything. Writes two arm dirs that vans_eval_metrics.py scores like any
generated arm:

  copy_last   the last input frame held still for 33 frames (ground/<id>_last.jpg from ground_objects.py,
              extracted again if missing)
  copy_input  the input clip itself, symlinked, as if nothing changes

Compare with VANS's own captions (CLIP-V 0.775), `nocot` captions (0.781) and GT captions (0.806), all
rendered by the released VANS video model.
"""
import argparse
import json
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    last_dir, input_dir = (os.path.join(args.out_dir, a, "videos") for a in ("copy_last", "copy_input"))
    os.makedirs(last_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    for k, r in enumerate(rows):
        sid, src = r["sample_id"], os.path.join(args.bench_dir, r["input_video"])
        frame = os.path.join(args.bench_dir, "ground", f"{sid}_last.jpg")
        if not os.path.exists(frame):
            frame = os.path.join(last_dir, f"{sid}_last.jpg")
            subprocess.run([args.ffmpeg, "-v", "error", "-y", "-sseof", "-1", "-i", src, "-update", "1", "-q:v", "2",
                            frame], check=True)
        dest = os.path.join(last_dir, f"{sid}.mp4")
        if not os.path.exists(dest):
            subprocess.run([args.ffmpeg, "-v", "error", "-y", "-loop", "1", "-framerate", "16", "-i", frame,
                            "-frames:v", "33", "-vf", "scale=640:352:force_original_aspect_ratio=increase,crop=640:352",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", dest], check=True)
        link = os.path.join(input_dir, f"{sid}.mp4")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(src), link)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(rows)}", flush=True)
    print(f"copy baselines written for {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()

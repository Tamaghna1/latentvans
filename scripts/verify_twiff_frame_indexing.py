"""Check twiff_frames.read_twiff_frames against TwiFF's own frames.

TwiFF's validation parquet embeds the actual JPEG for every question/reasoning
frame (question_images / reasoning_images columns). For each row whose cut clip
exists under --clip_dir, this decodes the requested indices with
read_twiff_frames and compares each one to the embedded JPEG (grayscale 64x36
MSE). It also scores the old raw-frame-number reading, and a random frame of the
same clip as the "unrelated frame" reference level.

A frame counts as matched when its error is below 10% of the random-frame error.

Usage:
  python verify_twiff_frame_indexing.py --val_parquet val-00000-of-00005.parquet \\
      --clip_dir $SCRATCH/data/panda70m_clips_val --max_rows 100
"""
import argparse
import io
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twiff_frames import read_twiff_frames  # noqa: E402


def small(arr_or_img):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    return np.asarray(img.convert("L").resize((64, 36)), dtype=np.float32)


def raw_frame(video_path, frame_no):
    """The old (wrong) reading: index k taken as 0-based frame number k."""
    import subprocess
    from twiff_frames import ffmpeg_exe

    out = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-threads", "1", "-i", video_path, "-vf",
         f"select=eq(n\\,{frame_no}),scale=64:36", "-frames:v", "1", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"], capture_output=True, timeout=120,
    )
    if len(out.stdout) < 64 * 36:
        return None
    return np.frombuffer(out.stdout[:64 * 36], np.uint8).reshape(36, 64).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_parquet", required=True, nargs="+")
    p.add_argument("--clip_dir", required=True)
    p.add_argument("--max_rows", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    n_rows = 0
    new_ok = old_ok = total = 0
    for path in args.val_parquet:
        table = pq.read_table(path, columns=["video", "question_images", "reasoning_images",
                                             "question_images_index", "reasoning_images_index"])
        for row in table.to_pylist():
            if n_rows >= args.max_rows:
                break
            clip = os.path.join(args.clip_dir, row["video"])
            if not os.path.exists(clip):
                continue
            idxs = row["question_images_index"] + row["reasoning_images_index"]
            imgs = row["question_images"] + row["reasoning_images"]
            try:
                got = read_twiff_frames(clip, idxs + [rng.randint(1, 8)])
            except Exception as e:
                print(f"skip {row['video']}: {e}")
                continue
            n_rows += 1
            for k, img in zip(idxs, imgs):
                gt = small(Image.open(io.BytesIO(img["bytes"])))
                err_new = float(((small(got[k]) - gt) ** 2).mean())
                rand_k = rng.choice([j for j in range(1, 9) if abs(j - k) > 1] or [k])
                rand_frame = read_twiff_frames(clip, [rand_k])[rand_k]
                err_rand = float(((small(rand_frame) - gt) ** 2).mean())
                old = raw_frame(clip, k)
                err_old = float(((old - gt) ** 2).mean()) if old is not None else float("nan")
                ok_new = err_new < 0.1 * err_rand
                ok_old = err_old < 0.1 * err_rand
                new_ok += ok_new
                old_ok += ok_old
                total += 1
                print(f"{row['video']:<28} idx={k} err_new={err_new:8.1f} err_old_rawidx={err_old:8.1f} "
                      f"err_random={err_rand:8.1f}  new={'OK' if ok_new else 'MISS'} old={'OK' if ok_old else 'MISS'}")
        if n_rows >= args.max_rows:
            break

    print("=" * 70)
    print(f"rows checked: {n_rows}, frames checked: {total}")
    print(f"read_twiff_frames matched: {new_ok}/{total} ({100 * new_ok / max(total, 1):.1f}%)")
    print(f"old raw-frame-number reading matched: {old_ok}/{total} ({100 * old_ok / max(total, 1):.1f}%)")
    print("(index 1 is the first frame under both readings, so the old reading always gets those right)")


if __name__ == "__main__":
    main()

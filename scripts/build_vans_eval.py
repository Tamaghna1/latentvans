"""Build a held-out VANS procedural benchmark from the released VANS-Data-100K CSVs.

VANS (arXiv 2511.16669) evaluates on 400 procedural + 400 predictive samples but did not
release that split. Only the procedural CSVs (COIN, YouCook2) are public, so we rebuild the
procedural half: one row per source video, drawn only from the official held-out subsets
(YouCook2 `validation`, COIN `testing`), so no eval video can appear in training rows taken
from the `training` subsets.

For each sampled row we yt-dlp just the span [input segment start, target segment end] of the
source video, then re-encode the input and target segments into separate mp4s. Rows whose video
is gone are skipped, and sampling continues until --per_source rows succeed for each source.

--split train builds SFT/RL training data the same way from the official `training` subsets
instead, with up to --rows_per_video rows per video and smaller re-encodes (crf 23).

Output: <out_dir>/{clips/<source>_<vid>_<id>.mp4, benchmark.jsonl, build_report.jsonl}
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import zlib

import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def segment_lookup(source, db, vid, seg_id):
    if source == "youcook":
        for a in db[vid]["annotations"]:
            if int(a["id"]) == int(seg_id):
                return a["segment"]
    else:
        for a in db[vid]["annotation"]:
            if str(a["id"]) == str(seg_id):
                return a["segment"]
    return None


def download_span(vid, start, end, dest, cookies, timeout):
    cmd = [
        "yt-dlp", f"https://www.youtube.com/watch?v={vid}",
        "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best",
        "--download-sections", f"*{max(0, start - 1):.2f}-{end + 1:.2f}",
        "--force-keyframes-at-cuts",
        "-o", dest, "--no-progress", "--quiet", "--no-warnings",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return r.returncode == 0, r.stderr.strip()[-300:]


def cut(src, offset, start, end, dest, crf):
    # re-encode (no stream copy) so the cut has no hidden pre-roll frames
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start - offset:.3f}", "-i", src,
           "-t", f"{end - start:.3f}", "-an", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", dest]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_source", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cookies", default=None)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--split", choices=["eval", "train"], default="eval")
    p.add_argument("--rows_per_video", type=int, default=1, help="--split train: max rows per source video")
    p.add_argument("--shard", type=int, default=0, help="parallel builders: this one takes videos with crc32(vid) %% num_shards == shard")
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    clips = os.path.join(args.out_dir, "clips")
    os.makedirs(clips, exist_ok=True)
    yc = json.load(open(os.path.join(args.meta_dir, "youcookii_annotations_trainval.json")))["database"]
    co = json.load(open(os.path.join(args.meta_dir, "COIN.json")))["database"]
    if args.split == "eval":
        sources = [("youcook", "YouCook.csv", yc, "validation"), ("coin", "COIN.csv", co, "testing")]
    else:
        sources = [("youcook", "YouCook.csv", yc, "training"), ("coin", "COIN.csv", co, "training")]
    crf = 18 if args.split == "eval" else 23

    bench_path = os.path.join(args.out_dir, "benchmark.jsonl")
    done = {}
    if os.path.exists(bench_path):
        for line in open(bench_path):
            row = json.loads(line)
            done[row["sample_id"]] = row
    report = open(os.path.join(args.out_dir, "build_report.jsonl"), "a")
    bench = open(bench_path, "a")

    for source, csv_name, db, subset in sources:
        df = pd.read_csv(os.path.join(args.meta_dir, csv_name))
        df["vid"] = df.input_video_path.str.split("/").str[1]
        df = df[df.vid.map(lambda v: db.get(v, {}).get("subset") == subset)]
        # videos in a fixed random order, rows within a video chosen with the same seeded rng
        rng = random.Random(args.seed)
        groups = sorted(df.groupby("vid").groups.items())
        rng.shuffle(groups)
        mine = lambda v: zlib.crc32(v.encode()) % args.num_shards == args.shard  # noqa: E731
        groups = [g for g in groups if mine(g[0])]
        quota = args.per_source // args.num_shards + (args.shard < args.per_source % args.num_shards)
        n_ok = sum(1 for r in done.values() if r["source"] == source and mine(r["vid"]))
        log(f"{source}: {len(groups)} {subset} videos, {n_ok} already built")
        for vid, idx in groups:
            if n_ok >= quota:
                break
            if args.split == "eval":
                picks = [sorted(idx)[rng.randrange(len(idx))]]  # one row per held-out video
            else:
                picks = sorted(idx)
                rng.shuffle(picks)
            for j in picks[:args.rows_per_video]:
                if n_ok >= quota:
                    break
                row = df.loc[j]
                sid = f"{source}_{vid}_{row.input_id}"
                if sid in done:
                    continue
                n_ok += build_row(args, source, db, vid, row, sid, clips, crf, report, bench, done)
        log(f"{source}: finished with {n_ok} samples")


def build_row(args, source, db, vid, row, sid, clips, crf, report, bench, done):
    """Download + cut one row; returns 1 on success, 0 otherwise."""
    in_seg = segment_lookup(source, db, vid, row.input_id)
    out_seg = segment_lookup(source, db, vid, row.target_id)
    status = {"sample_id": sid, "source": source, "vid": vid}
    if in_seg is None or out_seg is None:
        status["error"] = "segment not found"
        report.write(json.dumps(status) + "\n"); report.flush()
        return 0
    span_start, span_end = min(in_seg[0], out_seg[0]), max(in_seg[1], out_seg[1])
    tmp = tempfile.mkdtemp(dir=args.out_dir)
    try:
        ok, err = download_span(vid, span_start, span_end, os.path.join(tmp, "src.%(ext)s"),
                                args.cookies, args.timeout)
        files = [f for f in os.listdir(tmp) if f.startswith("src.")]
        if not ok or not files:
            status["error"] = f"download: {err}"
            report.write(json.dumps(status) + "\n"); report.flush()
            return 0
        src = os.path.join(tmp, files[0])
        offset = max(0, span_start - 1)
        in_path = os.path.join(clips, f"{sid}_in.mp4")
        out_path = os.path.join(clips, f"{sid}_out.mp4")
        if not (cut(src, offset, *in_seg, in_path, crf) and cut(src, offset, *out_seg, out_path, crf)):
            status["error"] = "cut failed"
            report.write(json.dumps(status) + "\n"); report.flush()
            return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    entry = {
        "sample_id": sid, "source": source, "vid": vid, "task": row.recipe_type,
        "input_video": os.path.relpath(in_path, args.out_dir),
        "target_video": os.path.relpath(out_path, args.out_dir),
        "input_segment": in_seg, "target_segment": out_seg,
        "input_sentence": row.input_sentence, "target_sentence": row.target_sentence,
        "instruction": row.ENG_Instruction, "think": row.ENG_Think,
        "gt_caption": str(row.ENG_GT_Caption).split("\n\n---")[0].strip(),
    }
    bench.write(json.dumps(entry) + "\n"); bench.flush()
    report.write(json.dumps({**status, "ok": True}) + "\n"); report.flush()
    done[sid] = entry
    log(f"{source}: {sum(1 for r in done.values() if r['source'] == source)}/{args.per_source} ({sid})")
    return 1


if __name__ == "__main__":
    main()

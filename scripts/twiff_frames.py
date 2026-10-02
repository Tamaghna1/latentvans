"""TwiFF frame-index convention, and a decord-free way to read those frames.

TwiFF-2.7M / Future-L1-50K's `question_images_index` / `reasoning_images_index`
are NOT raw video frame numbers. They index into 8 frames TwiFF sampled from
each cut clip (1-based):

    index 1      -> first frame
    index 2..7   -> centres of 6 equal segments, i.e. fraction (2k-3)/12
    index 8      -> last frame

so the 8 frames sit at fractions 0, 1/12, 3/12, 5/12, 7/12, 9/12, 11/12, 1.

Established 2026-10-02 by matching the JPEGs embedded in the TwiFF validation
parquet (question_images / reasoning_images) against every decoded frame of
the corresponding cut clip in panda70m_clips_val (scripts/legacy-free probe,
see probe_twiff_frame_indexing.py): across ~50 clips the best-matching frame
landed on this formula to within a few frames, e.g. a 1152-frame clip gave
indices 1..8 -> frames 1, 97, 288, 479, 671, 862, 1053, 1150 (formula: 0, 96,
288, 480, 672, 864, 1056, 1151). Reading index k as raw frame k instead (what
every script did before this date) returns frames from the first ~0.3 s of the
clip, so "context" and "future" frames were ~0.1 s apart.

Decoding goes through the ffmpeg CLI, single-threaded, instead of decord:
decord's threaded decoder rejects a large share of these stream-copied clips
(avcodec_send_packet EAGAIN / "cannot find video stream"), the ~35-40% skip
rate seen in every Stage 0 run, while ffmpeg decodes them cleanly.
"""
import os
import shutil
import subprocess

import numpy as np

TWIFF_NUM_FRAMES = 8


def twiff_index_to_frame(index, n_frames):
    """Map a 1-based TwiFF frame index to a 0-based frame number in a clip of n_frames."""
    if not 1 <= index <= TWIFF_NUM_FRAMES:
        raise ValueError(f"TwiFF frame index must be in 1..{TWIFF_NUM_FRAMES}, got {index}")
    if n_frames <= 0:
        raise ValueError(f"clip has no frames (n_frames={n_frames})")
    if index == 1:
        frame = 0
    elif index == TWIFF_NUM_FRAMES:
        frame = n_frames - 1
    else:
        frame = int((2 * index - 3) / 12 * n_frames)
    return min(max(frame, 0), n_frames - 1)


def ffmpeg_exe():
    """ffmpeg from the active conda env if present, else whatever is on PATH."""
    import sys

    env_ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if os.path.exists(env_ffmpeg):
        return env_ffmpeg
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("no ffmpeg binary in the active env's bin/ or on PATH")
    return found


def _probe_size(video_path, ffprobe):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=120,
    )
    fields = out.stdout.strip().splitlines()[0].split(",") if out.stdout.strip() else []
    if out.returncode != 0 or len(fields) < 2:
        raise RuntimeError(f"ffprobe failed on {video_path}: {out.stderr.strip()[:300]}")
    return int(fields[0]), int(fields[1])


def count_decoded_frames(video_path, timeout=300):
    """Number of frames ffmpeg actually outputs for the first video stream.

    Not the container's packet count: download_panda70m_clips.py's stream-copy
    cuts start on the keyframe before the requested start time and hide the
    pre-roll with an mp4 edit list, so a clip can carry tens of packets that
    are decoded but never displayed (e.g. 234 packets / 208 displayed frames).
    TwiFF's index positions are fractions of the displayed frames, so count those,
    with a tiny-resolution decode to keep this pass cheap.
    """
    out = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-threads", "1", "-i", video_path, "-map", "0:v:0",
         "-vf", "scale=16:8", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=timeout,
    )
    return len(out.stdout) // (16 * 8)


def read_twiff_frames(video_path, indices, timeout=300):
    """Decode video_path once and return {twiff_index: HxWx3 uint8 array} for the requested indices.

    Two single-threaded ffmpeg passes: count_decoded_frames() gets n, then a
    full-resolution decode streams raw RGB frames and keeps only the target ones,
    so memory stays at a few frames regardless of clip length. If the second pass
    yields fewer frames than counted, a target past the end falls back to the
    last decoded frame.
    """
    indices = sorted(set(int(i) for i in indices))
    ffmpeg = ffmpeg_exe()
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe") or ffprobe
    w, h = _probe_size(video_path, ffprobe)
    n = count_decoded_frames(video_path, timeout=timeout)
    if n <= 0:
        raise RuntimeError(f"ffmpeg decoded 0 frames from {video_path}")
    frame_bytes = w * h * 3
    wanted = {}
    for k in indices:
        wanted.setdefault(twiff_index_to_frame(k, n), []).append(k)
    last_needed = max(wanted)

    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-threads", "1", "-i", video_path, "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    out, last_buf, pos = {}, None, 0
    try:
        while pos <= last_needed:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            last_buf = buf
            for k in wanted.get(pos, ()):
                out[k] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            pos += 1
    finally:
        proc.kill()
        proc.wait(timeout=timeout)
    if last_buf is None:
        raise RuntimeError(f"ffmpeg decoded 0 frames from {video_path}")
    for k in indices:
        if k not in out:
            out[k] = np.frombuffer(last_buf, np.uint8).reshape(h, w, 3)
    return out

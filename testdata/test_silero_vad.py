#!/usr/bin/env python3
"""Compare the zilero-vad Zig implementation against the reference Silero VAD
ONNX model (official v6.2, 16 kHz, opset 15; CPU only, strictly 1 thread).

Subcommands:
  check   Run both implementations over every sample wav in this directory
          plus 5 minutes of deterministically generated audio, and compare
          the per-frame speech probabilities at two decimal positions.
  bench   Stream 5 minutes of generated audio through both implementations
          and report frames/s, realtime speed-up and peak RSS.

Usage (python is provided by uv; deps are declared inline below):
  uv run testdata/test_silero_vad.py check
  uv run testdata/test_silero_vad.py bench
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.17", "numpy"]
# ///

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# Strictly single-threaded: set before onnxruntime is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = ROOT / "model" / "silero_vad_16k_op15.onnx"
SR = 16000
FRAME = 512  # samples per frame (32 ms)
CONTEXT = 64  # last 64 samples of the previous frame, prepended to each call
GENERATED_SECONDS = 300  # 5 minutes
SEED = 1337


# --------------------------------------------------------------------------
# infrastructure
# --------------------------------------------------------------------------

def find_cli(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"error: cli not found: {p}")
        return p
    p = ROOT / "zig-out" / "bin" / "zilero-cli"
    if not p.exists():
        print(f"cli not found, building (ReleaseFast): {p}", file=sys.stderr)
        subprocess.run(["zig", "build", "-Doptimize=ReleaseFast"], cwd=ROOT, check=True)
    return p


def make_session() -> ort.InferenceSession:
    """ONNX session, CPU only, strictly 1 thread."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(MODEL_PATH), opts, providers=["CPUExecutionProvider"]
    )


def load_wav_i16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
            sys.exit(f"error: {path.name} is not 16 kHz mono 16-bit PCM")
        n = w.getnframes()
        return np.frombuffer(w.readframes(n), dtype=np.int16)


def i16_to_f32(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / np.float32(32768.0)


def n_frames(n_samples: int) -> int:
    return (n_samples + FRAME - 1) // FRAME


# --------------------------------------------------------------------------
# generated audio (deterministic, fixed seed)
# --------------------------------------------------------------------------

def generate_audio(seconds: int, seed: int = SEED) -> np.ndarray:
    """Return `seconds` of 16 kHz mono f32 audio as a single array.

    Content cycles every 4 s: 440 Hz sine (0.5 s), silence (0.5 s), white
    noise (0.75 s), 1 kHz AM tone (0.75 s), 200-2000 Hz chirp (1.0 s),
    silence (0.5 s). Fully deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    total = seconds * SR
    t = np.arange(total, dtype=np.float64) / SR
    noise = rng.standard_normal(total, dtype=np.float32)
    out = np.zeros(total, dtype=np.float32)

    cycle = [(0.5, 0), (0.5, 1), (0.75, 2), (0.75, 3), (1.0, 4), (0.5, 5)]
    pos = 0
    while pos < total:
        for dur, kind in cycle:
            e = min(pos + int(dur * SR), total)
            if e > pos:
                if kind == 0:  # 440 Hz sine
                    out[pos:e] = (0.5 * np.sin(2 * np.pi * 440 * t[pos:e])).astype(np.float32)
                elif kind == 2:  # white noise, -20 dBFS
                    out[pos:e] = (0.1 * noise[pos:e]).astype(np.float32)
                elif kind == 3:  # 1 kHz carrier, 5 Hz AM (speech-like modulation)
                    out[pos:e] = (0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t[pos:e]))
                                  * np.sin(2 * np.pi * 1000 * t[pos:e])).astype(np.float32)
                elif kind == 4:  # 200 -> 2000 Hz chirp
                    tc = t[pos:e] - t[pos]
                    out[pos:e] = (0.4 * np.sin(2 * np.pi * (200 * tc + 900 * tc * tc))).astype(np.float32)
                # kind 1 and 5: silence (already zero)
            pos = e
            if pos >= total:
                break
    return out


def to_i16(f32: np.ndarray) -> np.ndarray:
    return np.clip(np.round(f32 * 32768.0), -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# reference (ONNX) inference
# --------------------------------------------------------------------------

def reference_probs(f32: np.ndarray, session: ort.InferenceSession | None = None) -> np.ndarray:
    """Run the reference ONNX model frame by frame (512 samples, zero-padded
    tail) and return one probability per frame.

    Follows the official silero-vad OnnxWrapper protocol: each call receives
    `context ++ frame` (576 samples), where `context` is the last 64 samples
    of the previous call (zeros at stream start)."""
    sess = session or make_session()
    sr_in = np.array(SR, dtype=np.int64)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = np.empty(n_frames(len(f32)), dtype=np.float32)
    for n in range(n_frames(len(f32))):
        frame = f32[n * FRAME:(n + 1) * FRAME]
        if frame.size < FRAME:
            frame = np.pad(frame, (0, FRAME - frame.size))
        x = np.concatenate([context, frame])
        out, state = sess.run(
            ["output", "stateN"],
            {"input": x.reshape(1, FRAME + CONTEXT), "state": state, "sr": sr_in},
        )
        probs[n] = out[0][0]
        context = x[-CONTEXT:]
    return probs


# --------------------------------------------------------------------------
# zig CLI inference
# --------------------------------------------------------------------------

def cli_probs_lines(raw_i16: np.ndarray, cli: Path) -> list[str]:
    """Stream the raw i16 PCM through the CLI and return its stdout lines
    (simple mode: one probability per frame, two decimals)."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp.write(raw_i16.tobytes())
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            proc = subprocess.run(
                [str(cli)], stdin=f, stdout=subprocess.PIPE, check=True
            )
    finally:
        os.unlink(tmp_path)
    return proc.stdout.decode().splitlines()


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def check_one(name: str, f32: np.ndarray, session: ort.InferenceSession, cli: Path) -> bool:
    ref = reference_probs(f32, session)
    lines = cli_probs_lines(to_i16(f32), cli)

    ok = True
    if len(lines) != len(ref):
        print(f"  {name}: FAIL frame count mismatch (reference {len(ref)}, cli {len(lines)})")
        return False

    mismatches = []
    max_diff = 0.0
    for i, (p, line) in enumerate(zip(ref, lines)):
        ref2 = f"{float(p):.2f}"
        diff = abs(float(p) - float(line))
        max_diff = max(max_diff, diff)
        if ref2 != line:
            mismatches.append((i, ref2, line))

    print(f"  {name}: {len(ref)} frames, max |diff| = {max_diff:.6f}, "
          f"2-dec mismatches = {len(mismatches)}")
    for i, ref2, line in mismatches[:10]:
        print(f"    frame {i}: reference {ref2} vs cli {line}")
    if mismatches:
        ok = False
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_check(args: argparse.Namespace) -> int:
    cli = find_cli(args.cli)
    session = make_session()
    print(f"reference : {MODEL_PATH.name} (onnxruntime, 1 thread)")
    print(f"zilero    : {cli}")
    print()

    all_ok = True
    for wav in sorted(HERE.glob("*.wav")):
        print(f"sample: {wav.name}")
        f32 = i16_to_f32(load_wav_i16(wav))
        all_ok &= check_one(wav.name, f32, session, cli)
        print()

    print(f"generated: {GENERATED_SECONDS} s of synthetic audio (seed {SEED})")
    f32 = generate_audio(GENERATED_SECONDS)
    # Round-trip through i16 so the reference sees exactly the samples the
    # CLI receives (the comparison is of implementations, not of audio
    # quantization).
    f32 = i16_to_f32(to_i16(f32))
    all_ok &= check_one(f"generated_{GENERATED_SECONDS}s", f32, session, cli)

    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------

def peak_rss_sampler(pid: int) -> tuple[threading.Thread, list[int]]:
    """Sample VmHWM (peak RSS, kB) of a child process every 10 ms."""
    peak: list[int] = [0]
    state = {"stop": False}

    def run() -> None:
        status = f"/proc/{pid}/status"
        while not state["stop"]:
            try:
                with open(status) as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            peak[0] = max(peak[0], int(line.split()[1]))
                            break
            except OSError:
                break
            time.sleep(0.01)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, peak


def run_child_bench(cmd: list[str], raw_path: Path) -> tuple[float, int, float]:
    """Run a child that consumes the raw stream on stdin; return
    (wall_seconds, frames, peak_rss_mb).

    communicate() writes the stream to stdin while draining stdout, so a
    chatty child can never deadlock the pipes."""
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    sampler, peak = peak_rss_sampler(proc.pid)
    data = raw_path.read_bytes()
    t0 = time.perf_counter()
    proc.communicate(input=data)
    wall = time.perf_counter() - t0
    sampler.join(timeout=1)
    if proc.returncode != 0:
        sys.exit(f"error: bench child failed (exit {proc.returncode}): {cmd[0]}")
    return wall, n_frames(raw_path.stat().st_size // 2), peak[0] / 1024.0


def cmd_bench_reference(raw_path: str) -> None:
    """Hidden subcommand: reference ONNX bench child. Reads the raw i16 PCM
    stream from stdin, prints JSON {frames, wall_s}."""
    sess = make_session()
    sr_in = np.array(SR, dtype=np.int64)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    # Warm up the graph (excluded from the measured window).
    zeros = np.zeros((1, FRAME + CONTEXT), dtype=np.float32)
    for _ in range(64):
        sess.run(["output", "stateN"], {"input": zeros, "state": state, "sr": sr_in})

    frames = 0
    t0 = time.perf_counter()
    buf = b""

    def run_frame(frame: np.ndarray) -> None:
        nonlocal frames, state, context
        x = np.concatenate([context, frame])
        out, state = sess.run(
            ["output", "stateN"],
            {"input": x.reshape(1, FRAME + CONTEXT), "state": state, "sr": sr_in},
        )
        context = x[-CONTEXT:]
        frames += 1

    while True:
        chunk = sys.stdin.buffer.read(64 * 1024)
        if not chunk:
            break
        buf += chunk
        while len(buf) >= FRAME * 2:
            frame = np.frombuffer(buf[: FRAME * 2], dtype=np.int16).astype(np.float32)
            frame /= np.float32(32768.0)
            buf = buf[FRAME * 2:]
            run_frame(frame)
    if buf:  # final partial frame, zero-padded (same rule as the CLI)
        n = len(buf) // 2
        frame = np.frombuffer(buf[: n * 2], dtype=np.int16).astype(np.float32)
        frame /= np.float32(32768.0)
        frame = np.pad(frame, (0, FRAME - frame.size))
        run_frame(frame)
    wall = time.perf_counter() - t0
    print(json.dumps({"frames": frames, "wall_s": wall}))


def parse_cli_arg(arg: str) -> tuple[str, Path]:
    """Parse a `--cli` value: `[label=]path`."""
    label, sep, path = arg.partition("=")
    if not sep:
        label, path = Path(arg).name, arg
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: cli not found: {p}")
    return label, p


def cmd_bench(args: argparse.Namespace) -> int:
    if args.cli:
        clis = [parse_cli_arg(a) for a in args.cli]
    else:
        clis = [("zilero-cli (ReleaseFast)", find_cli(None))]
    seconds = args.seconds
    f32 = generate_audio(seconds)
    raw = to_i16(f32)
    nfr = n_frames(len(raw))
    print(f"bench: {seconds} s of generated audio (seed {SEED}), {nfr} frames, "
          f"16 kHz mono, CPU, 1 thread")
    print()

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp.write(raw.tobytes())
        raw_path = Path(tmp.name)
    try:
        # Reference: python child running the ONNX model (same stream protocol).
        ref_cmd = [sys.executable, str(Path(__file__).resolve()),
                   "_bench-reference", str(raw_path)]
        wall_ref, frames_ref, rss_ref = run_child_bench(ref_cmd, raw_path)

        # Zilero: each CLI build streaming the same bytes.
        rows = [("silero-vad onnx (1 thread)", frames_ref, wall_ref, rss_ref)]
        for label, cli in clis:
            wall, frames, rss = run_child_bench([str(cli)], raw_path)
            rows.append((label, frames, wall, rss))
    finally:
        os.unlink(raw_path)

    rt = seconds
    print(f"{'solution':28s} {'frames/s':>10s} {'realtime':>10s} {'peak RSS':>10s}")
    for name, frames, wall, rss in rows:
        print(f"{name:28s} {frames / wall:10.1f} {rt / wall:9.1f}x {rss:9.1f} MB")
    print()
    print("notes: reference wall time includes stdin streaming; ONNX session")
    print("warm-up (64 frames) excluded. peak RSS sampled from /proc (VmHWM).")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="compare per-frame probabilities (2 decimals)")
    p_check.add_argument("--cli", help="path to zilero-cli (default: build if needed)")
    p_check.set_defaults(fn=cmd_check)

    p_bench = sub.add_parser("bench", help="benchmark both implementations")
    p_bench.add_argument("--cli", action="append",
                         help="[label=]path to a zilero-cli build; repeatable "
                              "(default: build and bench the default ReleaseFast cli)")
    p_bench.add_argument("--seconds", type=int, default=GENERATED_SECONDS,
                         help=f"bench duration in seconds (default {GENERATED_SECONDS})")
    p_bench.set_defaults(fn=cmd_bench)

    p_ref = sub.add_parser("_bench-reference", help=argparse.SUPPRESS)
    p_ref.add_argument("raw_path")
    p_ref.set_defaults(fn=lambda a: (cmd_bench_reference(a.raw_path), 0)[1])

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
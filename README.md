# zilero-vad

A pure Zig 0.16 port of the [Silero VAD](https://github.com/snakers4/silero-vad)
(16 kHz mode): a small, accurate voice activity detector. No dependencies, no
libc, no heap allocations in the library (all scratch is stack), model weights
embedded at build time, hot loops SIMD-compiled via `@Vector` (AVX2 on
x86_64-v3, NEON on aarch64).

## Build

```sh
zig build -Doptimize=ReleaseFast
```

On x86_64 add `-Dcpu=x86_64_v3` to enable AVX2/FMA (otherwise the same code
compiles to SSE2). On aarch64 nothing extra is needed — NEON is the baseline.

Produces `zig-out/bin/zilero-cli`.

## Library usage

Add as a dependency in `build.zig.zon`:

```zig
.zig {
    .name = ".your_app",
    .dependencies = .{
        .zilero = .{ .path = "../zilero-vad" },
    },
}
```

```zig
const zilero = @import("zilero");

var vad: zilero.VAD = .{};
var frame: [zilero.frame_size]i16 = undefined; // 512 samples
// ... fill `frame` with 16 kHz mono PCM ...
const prob = vad.process_i16(&frame); // speech probability in [0, 1]
vad.reset(); // between unrelated streams
```

Contract: feed exactly `zilero.frame_size` (512) samples per call, 16 kHz
mono, little-endian `i16`. Call `reset()` before processing an unrelated
stream (LSTM state and convolution history are per-stream). The struct holds
all state (LSTM cells, conv history, STFT window) and is meant to live on the
caller's stack or in a long-lived object; `process` allocates nothing.

## CLI usage

```
simple voice activity detector cli. The cli read 16 bit signed 16Khz PCM samples from stdin and detect voice presence writing to stdout
options:
  -w  consider the input data as .wav file. (e.g.   cat file.wav | zilero-cli -w )
  -p  <prob 0-1> vad probability threshold (defaults: 0.55)
  -s  <min silence ms> detect segements of voice that have at least min silence between then. must be > 100ms.
output:
   simple mode: for each frame received will output the vad probability with two decimal positions (e.g. 1.00, 0.86, 0.12) to stdout.
   segment mode: if -s is provided will output voice segments jsonl with the follwing format: {"start":start_ms,"end":end_ms,"avg_prob":float}
```

Examples:

```sh
# one line per 32 ms frame: the speech probability, two decimals (0.00-1.00)
cat audio.raw | ./zig-out/bin/zilero-cli

# same, but the input is a 16-bit mono 16 kHz .wav file
cat audio.wav | ./zig-out/bin/zilero-cli -w

# emit speech segments (jsonl) with at least 500 ms of silence between them
cat audio.raw | ./zig-out/bin/zilero-cli -s 500
```

## How the weights are embedded

`tools/gen_weights.zig` runs at build time (host) on
`model/silero_vad_16k_op15.onnx` — the official Silero VAD v6.2 model (16
kHz, opset 15, from `src/silero_vad/data/` in the upstream repository). It
walks the protobuf with a minimal varint/length-delimited reader (no
protobuf library), pulls the 15 graph initializers, validates their
dtypes/shapes, re-lays the convolution weights from `[out][in][k]` to
`[out][k][in]` (convolution-friendly), and emits `zig-cache/.../weights.zig`
— a plain Zig file of `pub const` arrays that the library imports as a
private module. The output is cached by a content digest of the model file,
so a build that does not touch the model does not re-run the generator. To
swap models, replace `model/silero_vad_16k_op15.onnx` and rebuild.

## Tests

```sh
zig build test                 # debug (assertions on)
zig build test -Dcpu=x86_64_v3 # simd path, x86_64
```

The suite covers the scalar/simd primitives (conv, LSTM cell, sigmoid,
tanh-via-sigmoid, STFT, reflect padding), the end-to-end pipeline
(determinism, silence, impulse response, partial frames, stateful continuity,
reset), and a naive scalar cross-check of the full `process` pipeline against
the embedded weights.

An end-to-end cross-check against the reference ONNX model runs via
`testdata/test_silero_vad.py` (needs `uv` for onnxruntime + numpy):

```sh
uv run testdata/test_silero_vad.py check   # per-frame probabilities, 2 decimals
uv run testdata/test_silero_vad.py bench   # frames/s, realtime, peak RSS
```

`check` streams every sample wav in `testdata/` plus 5 minutes of
deterministically generated audio through both implementations (ONNX
Runtime, CPU, 1 thread, and the Zig CLI) and compares the per-frame
probabilities at two decimal positions.

## License

MIT. The model weights are © the Silero Team and are distributed under the
MIT license.
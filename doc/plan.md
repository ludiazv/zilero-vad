# Zilero VAD — Implementation Plan (handover for the coding worker)

Goal: a pure-Zig (0.16, `std` only, no libc, no allocations) port of Silero VAD, 16 kHz mode only,
with model weights baked into the binary at build time from the upstream `.safetensors` file,
SIMD via `@Vector`, targeting x86_64 (AVX2) and aarch64 (NEON).

Reference: `https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/tinygrad_model.py`
(71 lines — read it once before starting; everything below is derived from it).

Work through the phases **in order**. Each phase ends with a "Done when" checklist. Do not start
the next phase until the current one passes. Commit at the end of every phase.

---

## 0. Ground rules that apply to every phase

* Zig **0.16**. Before using any `std` API you are not 100% sure about (file IO, stdin/stdout,
  `std.Build`), open the installed std source (`zig env` prints `lib_dir`; look in
  `<lib_dir>/std/`) and copy the real signature. Do not guess from memory: 0.15/0.16 changed IO
  (`std.Io.Writer`/`std.Io.Reader`, `std.fs.File.stdout()`, buffered writers with a caller-owned
  buffer that must be `flush()`ed).
* Always run `zig fmt .`, `zig build`, `zig build test` before committing.
* Follow the companion document `zilero-vad-zig-coding-rules.md`.

---

## 1. Repository skeleton

Create this layout (git repo named `zilero-vad`):

```
zilero-vad/
  .gitignore                 # .zig-cache/ zig-out/
  LICENSE                    # MIT (upstream Silero VAD is MIT; keep its copyright line)
  README.md
  build.zig
  build.zig.zon
  model/silero_vad_16k.safetensors      # copied verbatim from upstream repo
  tools/gen_weights.zig                 # build-time generator (host program)
  src/zilero.zig                        # THE library, single file (tests live at its bottom)
  src/cli.zig                           # zilero-cli
  .github/workflows/ci.yml
  .github/workflows/release.yml
```

Steps:
1. `git init`, add `.gitignore`.
2. Download the model file:
   `curl -L -o model/silero_vad_16k.safetensors https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_16k.safetensors`
   Check it is a few MB and starts with an 8-byte little-endian length followed by `{` (JSON).
3. `build.zig.zon` (minimal):
   ```zig
   .{
       .name = .zilero_vad,
       .version = "0.1.0",
       .fingerprint = 0x0, // run `zig build` once; the compiler prints the correct value, paste it
       .minimum_zig_version = "0.16.0",
       .dependencies = .{},
       .paths = .{ "build.zig", "build.zig.zon", "src", "tools", "model", "README.md", "LICENSE" },
   }
   ```

Done when: `git log` shows the initial commit with the layout above (empty `.zig` files are fine).

---

## 2. Understand the model (no code — read this carefully)

Per 512-sample frame (32 ms at 16 kHz), the network does:

| Stage | Op | Input shape | Output shape |
|---|---|---|---|
| context | prepend last 64 samples of previous frame | 512 | 576 |
| reflect pad | pad 64 on the right | 576 | 640 |
| `stft_conv` | Conv1d 1→258, k=256, stride=128, no pad, no bias | 640 | [258][4] |
| magnitude | `sqrt(re² + im²)`, re = ch 0..128, im = ch 129..257 | [258][4] | [129][4] |
| `conv1` + relu | 129→128, k3, s1, p1 | [129][4] | [128][4] |
| `conv2` + relu | 128→64, k3, s2, p1 | [128][4] | [64][2] |
| `conv3` + relu | 64→64, k3, s2, p1 | [64][2] | [64][1] |
| `conv4` + relu | 64→128, k3, s1, p1 | [64][1] | [128] |
| `lstm_cell` | LSTMCell 128→128, state (h, c) | [128] | h,c [128] |
| relu(h) | | [128] | [128] |
| `final_conv` + sigmoid | Conv1d 128→1, k1, with bias | [128] | 1 |
| mean over time | time length is 1 → identity | 1 | prob |

Details you must get exactly right:
* **Reflect pad** (PyTorch semantics, edge not repeated): for `k in 0..63`, `padded[576+k] = padded[574-k]`.
* **STFT windows**: window `t` (t = 0..3) is `padded[t*128 .. t*128+256]`.
  `out[c][t] = dot(stft_w[c][0..256], window_t)`.
* **Context update**: after processing, `context = frame[448..512]` (last 64 raw samples of the
  *current* frame). Initial context is all zeros.
* **LSTM** (tinygrad `LSTMCell`, PyTorch gate order i, f, g, o):
  ```
  gates[0..512] = W_ih · x + b_ih + W_hh · h + b_hh
  i = sigmoid(gates[0..128])   f = sigmoid(gates[128..256])
  g = tanh(gates[256..384])    o = sigmoid(gates[384..512])
  c' = f*c + i*g ;  h' = o*tanh(c')
  ```
* **Output**: `prob = sigmoid(dot(final_w[0..128], relu(h')) + final_b)`.
* Input samples are `f32` in `[-1, 1]`. `i16` input is converted as `x / 32768.0`.
* State that persists between frames: `h[128]`, `c[128]`, `context[64]`. `reset()` zeroes these.

Expected tensors in the safetensors file (names follow tinygrad attribute paths). Verify in
phase 3 by printing the header; if a name differs, adapt the generator's name table, nothing else.

| name | shape | notes |
|---|---|---|
| `stft_conv.weight` | [258,1,256] | |
| `conv1.weight` / `conv1.bias` | [128,129,3] / [128] | |
| `conv2.weight` / `conv2.bias` | [64,128,3] / [64] | |
| `conv3.weight` / `conv3.bias` | [64,64,3] / [64] | |
| `conv4.weight` / `conv4.bias` | [128,64,3] / [128] | |
| `lstm_cell.weight_ih` / `lstm_cell.bias_ih` | [512,128] / [512] | |
| `lstm_cell.weight_hh` / `lstm_cell.bias_hh` | [512,128] / [512] | |
| `final_conv.weight` / `final_conv.bias` | [1,128,1] / [1] | |

---

## 3. Build-time weight generator — `tools/gen_weights.zig`

A small host program: `gen_weights <in.safetensors> <out.zig>`. It runs during `zig build`.

Safetensors format: bytes `0..8` = `u64` little-endian header length `N`; bytes `8..8+N` = JSON
`{"tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [begin, end]}, "__metadata__": {...}}`;
data buffer starts at byte `8+N`; offsets are relative to the data buffer start.

Steps:
1. Read the whole file (this program *may* allocate — it is a build tool, not the library; use
   `std.heap.page_allocator` or a `std.heap.ArenaAllocator`).
2. Parse the header with `std.json.parseFromSlice(std.json.Value, ...)`. Skip `__metadata__`.
3. For each expected name from the table in phase 2, look it up; on missing name, print all
   names found in the file and exit with code 1 (clear build error). Assert `dtype == "F32"`
   (if it is `F16`, convert with `@as(f32, @floatCast(@as(f16, @bitCast(u16_le))))`; otherwise fail).
4. Read the floats as little-endian `u32` → `@bitCast` to `f32`.
5. **Re-layout conv weights** from `[out][in][k]` to `[out][k][in]` so the inner (vectorised)
   loop runs over contiguous `in` values:
   `dst[(o*K + k)*IN + i] = src[(o*IN + i)*K + k]`. Apply to conv1..conv4. Keep `stft_conv.weight`
   as `[258][256]`, LSTM as `[512][128]`, `final_conv.weight` as `[128]`.
6. Emit `out.zig` with one flat `pub const` per tensor, e.g.
   ```zig
   // Generated by tools/gen_weights.zig — do not edit.
   pub const stft_w: [258 * 256]f32 = .{ 0x1.2p-3, ... };
   pub const conv1_w: [128 * 3 * 129]f32 = .{ ... };   // layout [out][k][in]
   pub const conv1_b: [128]f32 = .{ ... };
   ... conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b,
   pub const lstm_w_ih: [512 * 128]f32 = .{ ... };
   pub const lstm_b_ih: [512]f32 = .{ ... };
   pub const lstm_w_hh: [512 * 128]f32 = .{ ... };
   pub const lstm_b_hh: [512]f32 = .{ ... };
   pub const final_w: [128]f32 = .{ ... };
   pub const final_b: f32 = ...;
   ```
   Print floats in **hex-float form** (`{x}` format specifier) so the round trip is exact.
   Put ~16 values per line. Use a buffered writer and `flush()` at the end.
7. Sanity-check locally: run the generator by hand, confirm the output compiles
   (`zig test` a one-liner that imports it) and that `final_b` and a few values are finite.

Done when: running the generator produces a `weights.zig` (~1–2 MB) that compiles.

---

## 4. `build.zig`

Steps to wire (verify each `std.Build` call against the 0.16 std source):
1. `const target = b.standardTargetOptions(.{}); const optimize = b.standardOptimizeOption(.{});`
2. Generator: `b.addExecutable(.{ .name = "gen_weights", .root_module = b.createModule(.{ .root_source_file = b.path("tools/gen_weights.zig"), .target = b.graph.host, .optimize = .ReleaseSafe }) })`.
   **Must be built for the host** (`b.graph.host`) so cross-compiling the library still works.
3. `const gen = b.addRunArtifact(gen_exe); gen.addFileArg(b.path("model/silero_vad_16k.safetensors")); const weights_zig = gen.addOutputFileArg("weights.zig");`
4. Library module: `const zilero_mod = b.createModule(.{ .root_source_file = b.path("src/zilero.zig"), .target = target, .optimize = optimize });`
   then `zilero_mod.addAnonymousImport("weights", .{ .root_source_file = weights_zig });`
   Expose it: `b.addModule("zilero", ...)`-style so other packages can `@import("zilero")`
   (use the same module object for both; check the 0.16 API for `addModule` vs `createModule`).
5. CLI: `b.addExecutable(.{ .name = "zilero-cli", .root_module = b.createModule(.{ .root_source_file = b.path("src/cli.zig"), .target, .optimize, .imports = &.{ .{ .name = "zilero", .module = zilero_mod } } }) })`; `b.installArtifact(cli)`.
   Set `.link_libc = false` (default) — never link libc.
6. Tests: `const tests = b.addTest(.{ .root_module = zilero_mod });` — the library module itself is
   the test root, so `test` blocks inside `src/zilero.zig` see the `weights` import and all private
   functions. `const test_step = b.step("test", "Run tests"); test_step.dependOn(&b.addRunArtifact(tests).step);`
7. A `run` step for the CLI (`b.step("run", ...)`) is nice-to-have; skip if it costs time.

SIMD note: baseline `x86_64` has no AVX2. Users/CI build with `-Dcpu=x86_64_v3` for AVX2.
Baseline `aarch64` always has NEON. Do not add custom target logic in `build.zig`.

Done when: `zig build` succeeds (with stub `src/*.zig` containing `const weights = @import("weights");` to prove the import works) and a change to the safetensors path triggers regeneration.

---

## 5. The library — `src/zilero.zig` (single file)

### 5.1 Public API (exactly this, nothing else public except the `VAD` type and constants)

```zig
pub const sample_rate = 16000;
pub const frame_size = 512;          // samples per process() call (32 ms)

pub const VAD = struct {
    // Persistent state only (zeroed by reset). 320 floats = 1.28 KB.
    h: [128]f32 = @splat(0),
    c: [128]f32 = @splat(0),
    context: [64]f32 = @splat(0),

    /// Speech probability in [0,1] for one 512-sample frame of f32 samples in [-1,1].
    pub fn process(self: *VAD, frame: *const [frame_size]f32) f32
    /// Same, for signed 16-bit PCM (converted as x / 32768).
    pub fn process_i16(self: *VAD, frame: *const [frame_size]i16) f32
    /// Clear h, c and context.
    pub fn reset(self: *VAD) void
};
```

* Client usage: `var vad: zilero.VAD = .{};` — that is the whole "allocation".
* All scratch memory lives on the stack of `process` (see 5.3): two arrays, 640 + 516 floats
  (4.6 KB). Nothing is heap-allocated, and nothing in the struct needs zeroing besides state.
* `process_i16` converts into a local `[512]f32` on the stack, then calls `process`.

### 5.2 SIMD primitives (private)

Zig lets a `@Vector` be any comptime length; LLVM splits vectors wider than the hardware
register (8 f32 on AVX2, 4 on NEON) into multiple instructions automatically. **Do not chunk
manually.** All lengths in this model are comptime constants, so use whole-array vectors:

```zig
fn dot(comptime n: usize, a: *const [n]f32, b: *const [n]f32) f32 {
    @setFloatMode(.optimized); // lets LLVM reassociate the @reduce into a SIMD tree reduction
    const V = @Vector(n, f32);
    const va: V = a.*;
    const vb: V = b.*;
    return @reduce(.Add, va * vb);
}
fn relu(comptime n: usize, x: *[n]f32) void {      // x.* = @max(v, zero)
fn sigmoid(comptime n: usize, x: *[n]f32) void {   // 1 / (1 + @exp(-v))
fn tanhv(comptime n: usize, x: *[n]f32) void {     // 2*sigmoid(2v) - 1
fn sigmoid1(x: f32) f32                            // scalar, for the final output
```
Rules and gotchas:
* `@setFloatMode(.optimized)` is required in `dot` (or at file scope): in strict mode a float
  `@reduce(.Add)` is emitted as an ordered scalar add chain, which kills the vectorisation.
* Lengths that are not a multiple of the register width (129, 258) are fine; LLVM widens them.
* The SIMD lowering is done by LLVM. On x86_64, Debug builds use Zig's self-hosted backend
  (correct but scalar). Build the CLI and any benchmark with `-Doptimize=ReleaseFast`.
* Verify once: `zig build-obj src/zilero.zig -O ReleaseFast -femit-asm` (with the weights import
  wired, or a scratch file containing only `dot`) and check that `vfmadd`/`vmulps` (x86) or
  `fmla`/`fmul` (aarch64) appear with a short reduction, not a 256-long scalar chain.
* Optional, only if profiling later shows the STFT stage dominating: rewrite `dot` as a loop over
  32-lane chunks with one `@Vector(32, f32)` accumulator plus a scalar tail. Not part of this plan.

### 5.3 `process` — step by step (all lengths comptime constants)

Scratch is two stack arrays used ping-pong (each stage reads one and writes the other):

```zig
var a: [640]f32 align(32) = undefined;   // padded → c1 → c3 → gates
var b: [516]f32 align(32) = undefined;   // mag → c2 → c4
const padded: *[640]f32     = &a;
const mag:    *[4][129]f32  = @ptrCast(b[0..516]);
const c1:     *[4][128]f32  = @ptrCast(a[0..512]);
const c2:     *[2][64]f32   = @ptrCast(b[0..128]);
const c3:     *[64]f32      = a[0..64];
const c4:     *[128]f32     = b[0..128];
const gates:  *[512]f32     = a[0..512];
```
Every stage fully writes its output before the next stage reads it, so no zeroing is needed.
There are **no zero-padding rows**: conv boundaries are handled by skipping taps (step 3).

1. `pad(self, frame, padded)` (private helper, so it can be unit-tested):
   `padded[0..64] = context; padded[64..576] = frame; for k in 0..64: padded[576+k] = padded[574-k]`.
2. STFT + magnitude, **kernel-outer / window-inner** so each 1 KB kernel row is read from memory
   once and applied to the four L1-resident windows:
   ```
   for f in 0..129:
     const wr = stft_w[f*256 ..][0..256];  const wi = stft_w[(129+f)*256 ..][0..256];
     for t in 0..4:
       win = padded[t*128 ..][0..256]
       re = dot(256, wr, win); im = dot(256, wi, win)
       mag[t][f] = @sqrt(re*re + im*im)
   ```
3. Generic k3 conv with padding=1, relu fused, boundary taps skipped at comptime:
   ```
   fn conv3(comptime IN, comptime OUT, comptime T_IN, comptime T_OUT, comptime stride,
            w: *const [OUT*3*IN]f32, b: *const [OUT]f32,
            src: *const [T_IN][IN]f32, dst: *[T_OUT][OUT]f32) void
     for o in 0..OUT:
       inline for t in 0..T_OUT:
         acc = b[o]
         inline for k in 0..3:
           const r = @as(isize, t*stride + k) - 1     // source row; -1 and T_IN are padding
           if (r >= 0 and r < T_IN) acc += dot(IN, w[(o*3+k)*IN ..][0..IN], &src[r])
         dst[t][o] = @max(acc, 0)
   ```
   Calls: `conv3(129,128, 4,4, 1, conv1_w, conv1_b, mag, c1)`,
   `conv3(128,64, 4,2, 2, ..., c1, c2)`, `conv3(64,64, 2,1, 2, ..., c2, @ptrCast(c3))`,
   `conv3(64,128, 1,1, 1, ..., @ptrCast(c3), @ptrCast(c4))`.
   Output lengths check: (T_IN + 2 − 3) / stride + 1 → 4, 2, 1, 1.
4. LSTM: `for j in 0..512: gates[j] = dot(128, w_ih[j*128..], c4) + b_ih[j] + dot(128, w_hh[j*128..], &self.h) + b_hh[j]`.
   Then `sigmoid` on `gates[0..128]`, `gates[128..256]`, `gates[384..512]`; `tanhv` on `gates[256..384]`.
   With `V128 = @Vector(128, f32)`: `c = f*c + i*g; h = o * tanh(c)` as vector expressions
   written back into `self.c`, `self.h`.
5. Output, no buffer: `return sigmoid1(@reduce(.Add, final_w_vec * @max(h_vec, zero)) + final_b)`.
   (Do **not** relu `self.h` in place — the un-relu'd `h` is the LSTM state for the next frame.)
6. `self.context = frame[448..512].*`.

Memory summary: struct 1.28 KB, stack scratch 4.6 KB, weights ≈1.2 MB read-only in `.rodata`
(≈1.2 MB streamed per frame — that is the dominant memory cost and is inherent to the model).

### 5.4 Numerical guard
No NaN checks inside the hot path. Guarantee determinism: no global state, no threads.

Done when: `zig build` compiles the library with `-Dcpu=x86_64_v3` and with `-Dtarget=aarch64-linux`,
and phase 6 tests pass.

---

## 6. Tests — `test` blocks at the bottom of `src/zilero.zig`

Zig compiles `test "..." {}` blocks only when the file is the test root; they cost nothing in the
library build and can call private functions directly. Put them after all code, in this order:

1. **refAllDecls**: `test { std.testing.refAllDecls(@This()); }` so every declaration is analyzed.
2. **dot_matches_scalar**: `dot(129, ...)` and `dot(256, ...)` on seeded random arrays vs a plain
   scalar loop → `expectApproxEqRel(..., 1e-5)` (covers non-multiple-of-8 lengths).
3. **sigmoid_tanh_reference**: `sigmoid`/`tanhv` on `[8]f32{-10,-1,-0.5,0,0.5,1,10,20}` vs
   `1/(1+std.math.exp(-x))` and `std.math.tanh(x)` → `expectApproxEqAbs(1e-6)`.
4. **reflect_pad**: call the private `pad` helper with a ramp frame (`frame[i] = i`) and a
   non-zero context into a local `[640]f32`; check `padded[0..64] == context`, `padded[64] == 0`,
   `padded[575] == 511`, and `padded[576+k] == padded[574-k]` for k = 0, 1, 63.
5. **conv3_boundary**: `conv3(2,1, 3,3, 1, ...)` with weights all 1, bias 0, input rows
   `{1,1},{2,2},{3,3}` → output `{6, 12, 10}` (proves the skipped-tap padding is correct;
   relu keeps it positive).
6. **weights_sane**: `weights.final_b` finite; first/last element of each weight array finite;
   array lengths match the phase 2 shapes.
7. **silence_is_not_speech**: 20 frames of zeros → every prob `< 0.1`.
8. **i16_matches_f32**: seeded random i16 frame (`std.Random.DefaultPrng`); `process_i16` on one
   VAD vs `process` on `x/32768.0` on another → `expectApproxEqAbs(1e-6)`.
9. **reset_restores_initial_state**: 10 noise frames, `reset()`, 5 frames; compare with a fresh
   VAD on the same 5 frames → bit-identical (`expectEqual`).
10. **state_carries_over**: the same frame fed twice gives a different prob the second time
   (proves LSTM/context state is used).
11. **full_scale_input_is_finite**: all `32767`, then all `-32768` → probs finite and in `[0,1]`.
12. **prob_in_range**: 100 frames of seeded noise at −20 dBFS → all probs in `[0,1]`.
13. **(optional, only if Python + torch are available) reference_match** — this one goes in a
    separate `src/reference_test.zig` (add a second `addTest` for it, importing `zilero`):
    `tools/gen_reference.py` runs upstream `silero_vad` on a synthetic 5 s signal (seeded noise
    bursts) and writes `test/reference_probs.txt` (one prob per line) and `test/reference_input.raw`
    (i16 LE). The Zig test `@embedFile`s both and asserts `|p_zig - p_ref| < 2e-3`.
    If Python is not available, skip it and say so in the README.

Done when: `zig build test` passes on the host (Debug) and with `-Doptimize=ReleaseFast` (LLVM SIMD path). On x86_64 also run `-Dcpu=x86_64_v3`.

---

## 7. CLI — `src/cli.zig`

Usage text (print exactly this on `-h`, on bad args, exit code 2):

```
simple voice activity detector cli. The cli read 16 bit signed 16Khz PCM samples from stdin and detect voice presence writing to stdout
options:
  -w  consider the input data as .wav file. (e.g.   cat file.wav | zilero-cli -w )
  -p  <prob 0-1> vad probability threshold (defaults: 0.55)
  -s  <min silence ms> detect segements of voice that have at least min silence between then. must be > 100ms.
output:
   simple mode: for each frame received will output '1' if vad is >= p or '0' < p to stdout.
   segment mode: if -s is provided will output voice segments jsonl with the follwing format: {"start":start_ms,"end":end_ms,"avg_prob":float}
```

Steps:
1. **Args**: iterate `std.process.args` (check the 0.16 way to get args without an allocator, e.g.
   `std.process.ArgIterator` / `argsWithAllocator` — the CLI *may* use a small `FixedBufferAllocator`
   if the API needs one; the library never allocates). Parse `-w`, `-p <f32>`, `-s <u32>`, `-h`.
   Validate: `0 <= p <= 1`; if `-s` given, `s > 100` else print usage + exit 2.
2. **IO**: stdin/stdout via `std.fs.File.stdin()/stdout()` with caller-provided buffers
   (e.g. `[64 * 1024]u8` for stdin, `[4096]u8` for stdout). Flush stdout before exit.
   Read with a "read exactly N bytes or until EOF" loop; never assume one `read` returns a full frame.
3. **WAV (-w)**: minimal RIFF parser on the stream: read 12-byte header (`RIFF`, size, `WAVE`),
   then iterate chunks (`id[4]`, `size u32 LE`): `fmt ` → require `audio_format == 1` (PCM),
   `channels == 1`, `sample_rate == 16000`, `bits == 16` (otherwise print an error to stderr, exit 1);
   `data` → start streaming samples from here (ignore the declared size if it is 0 or 0xFFFFFFFF,
   just read to EOF); any other chunk → skip `size` bytes (+1 if odd). Chunks may arrive before or
   after `fmt `; require `fmt ` before `data`.
4. **Frame loop**: read 1024 bytes → 512 `i16` LE (`std.mem.readInt(i16, ..., .little)` or `@bitCast`
   after a byteswap on big-endian; just use `readInt`). A final partial frame is zero-padded to 512
   and processed. `prob = vad.process_i16(&frame)`. Frame `n` covers `[n*32, (n+1)*32)` ms.
5. **Simple mode**: write `'1'` if `prob >= p` else `'0'`, followed by `'\n'`, per frame.
6. **Segment mode** (`-s`): state machine
   ```
   in_speech=false; seg_start_ms; last_speech_end_ms; sum_prob; n_prob
   per frame (start_ms = n*32, end_ms = start_ms+32):
     if prob >= p:
        if !in_speech { in_speech=true; seg_start_ms=start_ms; sum_prob=0; n_prob=0 }
        last_speech_end_ms=end_ms; sum_prob+=prob; n_prob+=1
     else if in_speech and (end_ms - last_speech_end_ms) >= s:
        emit; in_speech=false
   at EOF: if in_speech: emit
   emit → {"start":<seg_start_ms>,"end":<last_speech_end_ms>,"avg_prob":<sum_prob/n_prob, 3 decimals>}\n
   ```
   `start`/`end` are integers (ms); `avg_prob` printed with `{d:.3}`.
7. Exit code 0 on success, 1 on IO/format error (message to stderr), 2 on usage error.

Manual check: `sox -n -r 16000 -c 1 -b 16 -t raw - synth 2 sine 440 | ./zig-out/bin/zilero-cli` should print
~63 lines of `0`/`1` (63 = ceil(32000/512)). With a real speech wav and `-s 300`, segments should appear.

Done when: both modes work on a raw stream and on a wav; bad `-s 50` prints usage and exits 2.

---

## 8. README.md

Sections, in this order, short:
1. Title + one paragraph: what it is, pure Zig 0.16, no deps, no allocations, weights embedded, SIMD.
2. Build: `zig build -Doptimize=ReleaseFast` (+ `-Dcpu=x86_64_v3` note for AVX2; aarch64 needs nothing).
3. Library usage (10-line example): add as a dependency in `build.zig.zon`, `@import("zilero")`,
   `var vad: zilero.VAD = .{}; const p = vad.process_i16(&frame);`, `vad.reset()`.
   State the contract: 512-sample frames, 16 kHz mono, call `reset()` between unrelated streams.
4. CLI usage: paste the usage block + two example command lines.
5. How weights are embedded (one paragraph: `tools/gen_weights.zig` runs at build time on the
   safetensors file and emits `weights.zig`; regenerate by replacing the model file).
6. Tests: `zig build test`; mention the optional Python reference test.
7. License: MIT; model weights © Silero Team, MIT.

---

## 9. GitHub Actions

`.github/workflows/ci.yml` — on `push` and `pull_request`:
* `ubuntu-latest`, `mlugg/setup-zig@v2` with `version: 0.16.0` (pin the exact 0.16.x that exists).
* Steps: `zig fmt --check .`, `zig build test`, `zig build test -Dcpu=x86_64_v3`,
  cross-compile check: `zig build -Dtarget=aarch64-linux -Doptimize=ReleaseFast`.

`.github/workflows/release.yml` — on `push` of tags `v*`:
* Matrix over targets: `x86_64-linux -Dcpu=x86_64_v3`, `aarch64-linux`, `x86_64-macos -Dcpu=x86_64_v3`,
  `aarch64-macos`, `x86_64-windows -Dcpu=x86_64_v3`. All built on `ubuntu-latest` (Zig cross-compiles).
* Per target: `zig build -Doptimize=ReleaseFast -Dtarget=<t> [-Dcpu=...] --prefix out/<t>`, then
  `tar czf zilero-cli-<t>.tar.gz -C out/<t>/bin .` (`.zip` for windows).
* Final job: `softprops/action-gh-release@v2` uploading all archives; release name = tag.

Done when: pushing tag `v0.1.0` produces a GitHub release with 5 archives.

---

## 10. Final checklist before handing back

- [ ] `src/zilero.zig` exposes only `VAD` (with `process`, `process_i16`, `reset`) and the two constants.
- [ ] `grep -n "allocator\|Allocator\|alloc(" src/zilero.zig` returns nothing.
- [ ] No `link_libc`, no `@cImport` anywhere.
- [ ] `zig build test` green; `zig fmt --check .` clean.
- [ ] `zig build -Dtarget=aarch64-linux` and `-Dtarget=x86_64-linux -Dcpu=x86_64_v3` both build.
- [ ] README, LICENSE, both workflows present; tag `v0.1.0` pushed.

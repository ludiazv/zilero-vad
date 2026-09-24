//! Zilero VAD — a pure Zig port of the Silero VAD (16 kHz mode).
//!
//! The network processes 512-sample (32 ms) frames of mono f32 audio in
//! [-1, 1] and returns a speech probability in [0, 1]. All model weights are
//! baked into the binary at build time (`@import("weights")`, generated from
//! `model/silero_vad_16k_op15.onnx` by `tools/gen_weights.zig`). The
//! library never allocates: persistent state lives in the `VAD` struct
//! (1.25 KB) and all scratch memory lives on the stack of `process` (4.6 KB).
//!
//! ## Algorithm flow (per frame)
//!
//! 1. **Pad** — `context ++ frame ++ reflect(frame)` → 640 samples, where
//!    `context` is the last 64 samples of the previous frame (zeros at
//!    stream start).
//! 2. **STFT** — the reference model's 256-tap, hop-128 STFT, expressed as
//!    a conv1d over the padded buffer → magnitude spectrogram `[4][129]`.
//! 3. **Convs** — four k3 conv1d layers with fused relu, temporal strides
//!    1, 2, 2, 1: `[4][129] → [4][128] → [2][64] → [1][64] → [1][128]`.
//! 4. **LSTM** — one LSTMCell step (128 units, PyTorch gate order i, f, g,
//!    o) on the conv output; `h` and `c` persist between frames.
//! 5. **Output** — `sigmoid(w · relu(h) + b)` → speech probability.
//!
//! ## Data layout: transposed weights and features
//!
//! In the ONNX file, conv weights are stored PyTorch-style as
//! `[out][ci][k]` (input channel outer, kernel tap inner). The build-time
//! generator re-lays them out as `[out][k][ci]` — kernel tap outer, input
//! channel inner. Features are kept the same way: `[t][ci]` (time row
//! outer, channel inner) rather than `[ci][t]`. With both sides in
//! `[k][ci]` / `[t][ci]` form, every convolution tap over one time row is a
//! single contiguous dot product of two flat `f32` arrays — the innermost
//! loop of the whole network is `dot(IN, w_row, x_row)`, which lowers to
//! SIMD with no gathers, no runtime transposes, and no inner channel loop.
//!
//! ## Scratch memory
//!
//! `process` uses two `align(32)` buffers — 640 and 516 floats (4.6 KB) —
//! and ping-pongs them through the stages, reinterpreting each buffer's
//! bytes for the next stage's shape (padded→c1→c3→gates in one, mag→c2→c4
//! in the other). Every stage fully overwrites its output before the next
//! stage reads it, so no buffer is ever zeroed or copied.

const std = @import("std");
const weights = @import("weights");

/// Sampling rate of the input audio, in Hz.
pub const sample_rate = 16000;
/// Number of samples per `process` call (32 ms at 16 kHz).
pub const frame_size = 512;

/// Voice activity detector. Zero-initialized via `var vad: VAD = .{};`;
/// call `reset` between unrelated audio streams. Holds all persistent state
/// (1.25 KB); per-frame scratch (4.6 KB) lives on the stack of `process`.
pub const VAD = struct {
    /// LSTM hidden state, persists between frames.
    h: [128]f32 = @splat(0),
    /// LSTM cell state, persists between frames.
    c: [128]f32 = @splat(0),
    /// Last 64 raw samples of the previous frame; zeros for the first frame.
    context: [64]f32 = @splat(0),

    /// Speech probability in [0,1] for one 512-sample frame of f32 samples
    /// in [-1,1].
    pub fn process(self: *VAD, frame: *const [frame_size]f32) f32 {
        // Ping-pong scratch: every stage fully writes its output before the
        // next stage reads its input, so no zeroing is needed.
        var a: [640]f32 align(32) = undefined;
        var b: [516]f32 align(32) = undefined;
        const padded: *[640]f32 = &a;
        const mag: *[4][129]f32 = @ptrCast(&b);
        const c1: *[4][128]f32 = @ptrCast(&a);
        const c2: *[2][64]f32 = @ptrCast(&b);
        const c3: *[64]f32 = a[0..64];
        const c4: *[128]f32 = @ptrCast(&b);
        const gates: *[512]f32 = a[0..512];

        pad(self, frame, padded);
        stftMagnitude(padded, mag);
        conv3(129, 128, 4, 4, 1, &weights.conv1_w, &weights.conv1_b, mag, c1);
        conv3(128, 64, 4, 2, 2, &weights.conv2_w, &weights.conv2_b, c1, c2);
        conv3(64, 64, 2, 1, 2, &weights.conv3_w, &weights.conv3_b, c2, @ptrCast(c3));
        conv3(64, 128, 1, 1, 1, &weights.conv4_w, &weights.conv4_b, @ptrCast(c3), @ptrCast(c4));
        lstmStep(self, c4, gates);
        self.context = frame[448..512].*;
        return finalOutput(self);
    }

    /// Same as `process`, for signed 16-bit PCM (converted as x / 32768).
    pub fn process_i16(self: *VAD, frame: *const [frame_size]i16) f32 {
        var f: [frame_size]f32 = undefined;
        for (frame, 0..) |s, i| {
            f[i] = @as(f32, @floatFromInt(s)) / 32768.0;
        }
        return self.process(&f);
    }

    /// Clear h, c and context.
    pub fn reset(self: *VAD) void {
        self.h = @splat(0);
        self.c = @splat(0);
        self.context = @splat(0);
    }
};

/// SIMD dot product of two `n`-lane f32 arrays.
///
/// The whole product is one multiply over an `@Vector(n, f32)` plus a single
/// `@reduce(.Add)`: the compiler chunks the vector into the target's SIMD
/// lanes (SSE2 baseline, AVX2 with `-Dcpu=x86_64_v3`, NEON on aarch64) and
/// builds an FMA add tree for the reduction. `@setFloatMode(.optimized)` is
/// required — in strict mode the `@reduce` lowers to an ordered scalar add
/// chain, which is what this function exists to avoid.
fn dot(comptime n: usize, a: *const [n]f32, b: *const [n]f32) f32 {
    @setFloatMode(.optimized);
    const V = @Vector(n, f32);
    const va: V = a.*;
    const vb: V = b.*;
    return @reduce(.Add, va * vb);
}

/// Vector sigmoid: `1 / (1 + exp(-v))`.
fn sigmoidOf(comptime n: usize, v: @Vector(n, f32)) @Vector(n, f32) {
    const one: @Vector(n, f32) = @splat(1);
    const neg: @Vector(n, f32) = @splat(-1);
    return one / (one + @exp(v * neg));
}

/// Vector tanh via the identity `tanh(v) = 2·sigmoid(2v) − 1`. There is no
/// vector `@tanh` builtin; this reuses the vector sigmoid path and matches
/// `std.math.tanh` to f32 precision (see tests).
fn tanhOf(comptime n: usize, v: @Vector(n, f32)) @Vector(n, f32) {
    const two: @Vector(n, f32) = @splat(2);
    const one: @Vector(n, f32) = @splat(1);
    return two * sigmoidOf(n, two * v) - one;
}

fn relu(comptime n: usize, x: *[n]f32) void {
    const V = @Vector(n, f32);
    const v: V = x.*;
    x.* = @max(v, @as(V, @splat(0)));
}

fn sigmoid(comptime n: usize, x: *[n]f32) void {
    const V = @Vector(n, f32);
    const v: V = x.*;
    x.* = sigmoidOf(n, v);
}

fn tanhv(comptime n: usize, x: *[n]f32) void {
    const V = @Vector(n, f32);
    const v: V = x.*;
    x.* = tanhOf(n, v);
}

/// Scalar sigmoid, for the final output.
fn sigmoid1(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// Lays out `context ++ frame ++ reflect(frame)` into `padded` (640
/// floats). The reflect tail follows PyTorch semantics: the edge sample is
/// not repeated, so `padded[576 + k] = padded[574 - k]`.
fn pad(self: *VAD, frame: *const [frame_size]f32, padded: *[640]f32) void {
    @memcpy(padded[0..64], self.context[0..64]);
    @memcpy(padded[64..576], frame[0..512]);
    for (0..64) |k| {
        padded[576 + k] = padded[574 - k];
    }
}

/// STFT magnitude. The reference model implements the 256-tap, hop-128
/// STFT as a conv1d (`stft_conv`): `weights.stft_w` holds 258 rows of 256
/// taps — rows 0..128 the real part, rows 129..257 the imaginary part, one
/// row per frequency bin. Each of the four time windows is a contiguous
/// 256-float slice of `padded`, so each bin is two 256-lane dot products:
/// `mag[t][f] = hypot(dot(w_re[f], win), dot(w_im[f], win))`.
///
/// The loop is kernel-outer / window-inner: each 1 KB weight row is read
/// from memory once and applied to the four L1-resident windows.
fn stftMagnitude(padded: *const [640]f32, mag: *[4][129]f32) void {
    for (0..129) |f| {
        const wr: *const [256]f32 = weights.stft_w[f * 256 ..][0..256];
        const wi: *const [256]f32 = weights.stft_w[(129 + f) * 256 ..][0..256];
        for (0..4) |t| {
            const win: *const [256]f32 = padded[t * 128 ..][0..256];
            const re = dot(256, wr, win);
            const im = dot(256, wi, win);
            mag[t][f] = @sqrt(re * re + im * im);
        }
    }
}

/// Generic k3 conv1d (padding 1, fused relu) with comptime shape.
///
/// `w` is `[out][k][ci]` — the build-time transpose of the ONNX
/// `[out][ci][k]` layout — and `src` is `[t][ci]`, so each (tap, time) pair
/// is one contiguous `dot(IN, ...)` over the whole channel axis: no inner
/// channel loop, no gathers, no runtime transpose. Boundary taps (source
/// row −1 or T_IN) are skipped at comptime, so there are no zero-padding
/// rows and no runtime branch. `stride` is the temporal stride (1 or 2).
fn conv3(
    comptime IN: usize,
    comptime OUT: usize,
    comptime T_IN: usize,
    comptime T_OUT: usize,
    comptime stride: usize,
    w: *const [OUT * 3 * IN]f32,
    b: *const [OUT]f32,
    src: *const [T_IN][IN]f32,
    dst: *[T_OUT][OUT]f32,
) void {
    for (0..OUT) |o| {
        inline for (0..T_OUT) |t| { // Unrolled comptime loop over the input channels.
            var acc = b[o]; // Load the vias in the accumulator

            inline for (0..3) |k| { // inline for (unrolled at compile time) for 3 elementes of the kernel.

                const r = @as(isize, t * stride + k) - 1; // virtual zero pading this is comptime variable.
                if (r >= 0 and r < T_IN) { // This is a comptime if no overhad.
                    const row: *const [IN]f32 = &src[@as(usize, @intCast(r))];
                    const wr: *const [IN]f32 = w[(o * 3 + k) * IN ..][0..IN];
                    acc += dot(IN, wr, row);
                }
            }
            dst[t][o] = @max(acc, 0); // Fused relu
        }
    }
}

/// One LSTMCell step (128 units, PyTorch gate order i, f, g, o).
///
/// The gate matrices are row-major `[gate][ci]`, so all 512 pre-activations
/// are 128-lane dot products against `x` and the previous `h`
/// (`w_ih[j]·x + b_ih[j] + w_hh[j]·h + b_hh[j]`), written into `gates`
/// (512 floats of scratch). The non-linearities then run in place as vector
/// ops over four 128-lane chunks, and the state update is two vector
/// expressions: `c' = f·c + i·g`, `h' = o·tanh(c')`.
fn lstmStep(self: *VAD, x: *const [128]f32, gates: *[512]f32) void {
    for (0..512) |j| {
        const wr_ih: *const [128]f32 = weights.lstm_w_ih[j * 128 ..][0..128];
        const wr_hh: *const [128]f32 = weights.lstm_w_hh[j * 128 ..][0..128];
        gates[j] = dot(128, wr_ih, x) + weights.lstm_b_ih[j] + dot(128, wr_hh, &self.h) + weights.lstm_b_hh[j];
    }
    sigmoid(128, gates[0..128]);
    sigmoid(128, gates[128..256]);
    tanhv(128, gates[256..384]);
    sigmoid(128, gates[384..512]);

    const V = @Vector(128, f32);
    const i: V = (gates[0..128]).*;
    const f: V = (gates[128..256]).*;
    const g: V = (gates[256..384]).*;
    const o: V = (gates[384..512]).*;
    const c: V = self.c;
    const c_new = f * c + i * g;
    self.c = c_new;
    self.h = o * tanhOf(128, c_new);
}

/// Final layer: `sigmoid(w · relu(h) + b)`. The relu lives only in the
/// vector expression — `self.h` keeps the un-relu'd state for the next
/// frame, so it is not modified here.
fn finalOutput(self: *VAD) f32 {
    @setFloatMode(.optimized);
    const V = @Vector(128, f32);
    const h: V = self.h;
    const w: V = weights.final_w;
    const z = @reduce(.Add, w * @max(h, @as(V, @splat(0)))) + weights.final_b;
    return sigmoid1(z);
}
test {
    std.testing.refAllDecls(@This());
}

test "dot matches scalar loop" {
    var prng = std.Random.DefaultPrng.init(0x1234);
    var a: [256]f32 = undefined;
    var b: [256]f32 = undefined;
    for (0..256) |i| {
        a[i] = prng.random().float(f32) * 2 - 1;
        b[i] = prng.random().float(f32) * 2 - 1;
    }

    const d256 = dot(256, &a, &b);
    var s256: f32 = 0;
    for (0..256) |i| s256 += a[i] * b[i];
    try std.testing.expectApproxEqRel(s256, d256, 1e-5);

    var a129: [129]f32 = a[0..129].*;
    var b129: [129]f32 = b[0..129].*;
    const d129 = dot(129, &a129, &b129);
    var s129: f32 = 0;
    for (0..129) |i| s129 += a129[i] * b129[i];
    try std.testing.expectApproxEqRel(s129, d129, 1e-5);
}

test "sigmoid and tanh match reference" {
    const xs = [_]f32{ -10, -1, -0.5, 0, 0.5, 1, 10, 20 };

    var x: [8]f32 = xs;
    sigmoid(8, &x);
    for (xs, 0..) |e, i| {
        try std.testing.expectApproxEqAbs(1 / (1 + @exp(-e)), x[i], 1e-6);
    }

    var y: [8]f32 = xs;
    tanhv(8, &y);
    for (xs, 0..) |e, i| {
        try std.testing.expectApproxEqAbs(std.math.tanh(e), y[i], 1e-6);
    }
}

test "reflect pad matches pytorch semantics" {
    var vad: VAD = .{};
    for (0..64) |i| vad.context[i] = @as(f32, @floatFromInt(100 + i));
    var frame: [frame_size]f32 = undefined;
    for (0..frame_size) |i| frame[i] = @as(f32, @floatFromInt(i));

    var padded: [640]f32 = undefined;
    pad(&vad, &frame, &padded);

    for (0..64) |i| try std.testing.expectEqual(vad.context[i], padded[i]);
    try std.testing.expectEqual(@as(f32, 0), padded[64]);
    try std.testing.expectEqual(@as(f32, 511), padded[575]);
    inline for ([_]usize{ 0, 1, 63 }) |k| {
        try std.testing.expectEqual(padded[574 - k], padded[576 + k]);
    }
}

test "conv3 boundary taps are skipped" {
    const w: [1 * 3 * 2]f32 = @splat(1);
    const b: [1]f32 = @splat(0);
    const src: [3][2]f32 = .{ .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 } };
    var dst: [3][1]f32 = undefined;
    conv3(2, 1, 3, 3, 1, &w, &b, &src, &dst);
    try std.testing.expectEqual(@as(f32, 6), dst[0][0]);
    try std.testing.expectEqual(@as(f32, 12), dst[1][0]);
    try std.testing.expectEqual(@as(f32, 10), dst[2][0]);
}

test "weights are sane" {
    try std.testing.expect(std.math.isFinite(weights.final_b));
    const first_last = struct {
        fn check(arr: []const f32) !void {
            try std.testing.expect(std.math.isFinite(arr[0]));
            try std.testing.expect(std.math.isFinite(arr[arr.len - 1]));
        }
    }.check;
    try first_last(&weights.stft_w);
    try first_last(&weights.conv1_w);
    try first_last(&weights.conv1_b);
    try first_last(&weights.conv2_w);
    try first_last(&weights.conv2_b);
    try first_last(&weights.conv3_w);
    try first_last(&weights.conv3_b);
    try first_last(&weights.conv4_w);
    try first_last(&weights.conv4_b);
    try first_last(&weights.lstm_w_ih);
    try first_last(&weights.lstm_b_ih);
    try first_last(&weights.lstm_w_hh);
    try first_last(&weights.lstm_b_hh);
    try first_last(&weights.final_w);

    try std.testing.expectEqual(@as(usize, 258 * 256), weights.stft_w.len);
    try std.testing.expectEqual(@as(usize, 128 * 3 * 129), weights.conv1_w.len);
    try std.testing.expectEqual(@as(usize, 128), weights.conv1_b.len);
    try std.testing.expectEqual(@as(usize, 64 * 3 * 128), weights.conv2_w.len);
    try std.testing.expectEqual(@as(usize, 64), weights.conv2_b.len);
    try std.testing.expectEqual(@as(usize, 64 * 3 * 64), weights.conv3_w.len);
    try std.testing.expectEqual(@as(usize, 64), weights.conv3_b.len);
    try std.testing.expectEqual(@as(usize, 128 * 3 * 64), weights.conv4_w.len);
    try std.testing.expectEqual(@as(usize, 128), weights.conv4_b.len);
    try std.testing.expectEqual(@as(usize, 512 * 128), weights.lstm_w_ih.len);
    try std.testing.expectEqual(@as(usize, 512), weights.lstm_b_ih.len);
    try std.testing.expectEqual(@as(usize, 512 * 128), weights.lstm_w_hh.len);
    try std.testing.expectEqual(@as(usize, 512), weights.lstm_b_hh.len);
    try std.testing.expectEqual(@as(usize, 128), weights.final_w.len);
}

test "silence is not speech" {
    var vad: VAD = .{};
    const zero: [frame_size]f32 = @splat(0);
    for (0..20) |_| {
        const p = vad.process(&zero);
        try std.testing.expect(p < 0.1);
    }
}

test "i16 path matches f32 path" {
    var prng = std.Random.DefaultPrng.init(0x99);
    var frame_i16: [frame_size]i16 = undefined;
    var frame_f32: [frame_size]f32 = undefined;
    for (0..frame_size) |i| {
        const s = prng.random().intRangeAtMost(i16, -32768, 32767);
        frame_i16[i] = s;
        frame_f32[i] = @as(f32, @floatFromInt(s)) / 32768.0;
    }
    var v1: VAD = .{};
    var v2: VAD = .{};
    const p1 = v1.process_i16(&frame_i16);
    const p2 = v2.process(&frame_f32);
    try std.testing.expectApproxEqAbs(p2, p1, 1e-6);
}

test "reset restores initial state" {
    var prng = std.Random.DefaultPrng.init(0x42);
    var frames: [15][frame_size]f32 = undefined;
    for (&frames) |*f| {
        for (f) |*s| s.* = prng.random().float(f32) * 2 - 1;
    }

    var v1: VAD = .{};
    for (frames[0..10]) |f| _ = v1.process(&f);
    v1.reset();
    var probs1: [5]f32 = undefined;
    for (frames[10..15], 0..) |f, i| probs1[i] = v1.process(&f);

    var v2: VAD = .{};
    var probs2: [5]f32 = undefined;
    for (frames[10..15], 0..) |f, i| probs2[i] = v2.process(&f);

    try std.testing.expectEqual(probs1, probs2);
}

test "state carries over between frames" {
    var prng = std.Random.DefaultPrng.init(7);
    var frame: [frame_size]f32 = undefined;
    for (&frame) |*s| s.* = prng.random().float(f32) * 2 - 1;
    var vad: VAD = .{};
    const p1 = vad.process(&frame);
    const p2 = vad.process(&frame);
    try std.testing.expect(p1 != p2);
}

test "full scale input is finite" {
    var vad: VAD = .{};
    const pos: [frame_size]i16 = @splat(32767);
    const neg: [frame_size]i16 = @splat(-32768);
    for (0..5) |_| {
        const p = vad.process_i16(&pos);
        try std.testing.expect(std.math.isFinite(p));
        try std.testing.expect(p >= 0 and p <= 1);
    }
    for (0..5) |_| {
        const p = vad.process_i16(&neg);
        try std.testing.expect(std.math.isFinite(p));
        try std.testing.expect(p >= 0 and p <= 1);
    }
}

test "prob stays in range on noise" {
    var vad: VAD = .{};
    var prng = std.Random.DefaultPrng.init(0xbeef);
    var frame: [frame_size]f32 = undefined;
    for (0..100) |_| {
        // -20 dBFS noise: amplitude 0.1.
        for (&frame) |*s| s.* = prng.random().float(f32) * 0.2 - 0.1;
        const p = vad.process(&frame);
        try std.testing.expect(p >= 0 and p <= 1);
    }
}

/// Naive scalar re-implementation of the network, written directly from the
/// reference `tinygrad_model.py`. Uses scalar loops, `std.math.tanh` and a
/// different reduction order than the optimized pipeline; exists purely to
/// cross-check it. Both read the generator's [out][k][in] conv layout.
fn naiveProcess(
    context: *const [64]f32,
    frame: *const [frame_size]f32,
    h: *const [128]f32,
    c: *const [128]f32,
) f32 {
    var padded: [640]f32 = undefined;
    @memcpy(padded[0..64], context[0..64]);
    @memcpy(padded[64..576], frame[0..512]);
    for (0..64) |k| {
        padded[576 + k] = padded[574 - k];
    }

    var mag: [4][129]f32 = undefined;
    for (0..4) |t| {
        for (0..129) |f| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..256) |k| {
                re += weights.stft_w[f * 256 + k] * padded[t * 128 + k];
                im += weights.stft_w[(129 + f) * 256 + k] * padded[t * 128 + k];
            }
            mag[t][f] = @sqrt(re * re + im * im);
        }
    }

    // Conv1: 129 -> 128, k3 s1 p1. Original layout [out][in][k].
    var c1: [4][128]f32 = undefined;
    for (0..4) |t| {
        for (0..128) |o| {
            var acc = weights.conv1_b[o];
            for (0..3) |k| {
                const r = @as(isize, @intCast(t + k)) - 1;
                if (r >= 0 and r < 4) {
                    for (0..129) |i| {
                        acc += weights.conv1_w[(o * 3 + k) * 129 + i] * mag[@as(usize, @intCast(r))][i];
                    }
                }
            }
            c1[t][o] = @max(acc, 0);
        }
    }

    // Conv2: 128 -> 64, k3 s2 p1.
    var c2: [2][64]f32 = undefined;
    for (0..2) |t| {
        for (0..64) |o| {
            var acc = weights.conv2_b[o];
            for (0..3) |k| {
                const r = @as(isize, @intCast(t * 2 + k)) - 1;
                if (r >= 0 and r < 4) {
                    for (0..128) |i| {
                        acc += weights.conv2_w[(o * 3 + k) * 128 + i] * c1[@as(usize, @intCast(r))][i];
                    }
                }
            }
            c2[t][o] = @max(acc, 0);
        }
    }

    // Conv3: 64 -> 64, k3 s2 p1.
    var c3: [1][64]f32 = undefined;
    for (0..1) |t| {
        for (0..64) |o| {
            var acc = weights.conv3_b[o];
            for (0..3) |k| {
                const r = @as(isize, @intCast(t * 2 + k)) - 1;
                if (r >= 0 and r < 2) {
                    for (0..64) |i| {
                        acc += weights.conv3_w[(o * 3 + k) * 64 + i] * c2[@as(usize, @intCast(r))][i];
                    }
                }
            }
            c3[t][o] = @max(acc, 0);
        }
    }

    // Conv4: 64 -> 128, k3 s1 p1.
    var c4: [1][128]f32 = undefined;
    for (0..1) |t| {
        for (0..128) |o| {
            var acc = weights.conv4_b[o];
            for (0..3) |k| {
                const r = @as(isize, @intCast(t + k)) - 1;
                if (r >= 0 and r < 1) {
                    for (0..64) |i| {
                        acc += weights.conv4_w[(o * 3 + k) * 64 + i] * c3[@as(usize, @intCast(r))][i];
                    }
                }
            }
            c4[t][o] = @max(acc, 0);
        }
    }

    // LSTMCell: gate order i, f, g, o.
    var h2: [128]f32 = undefined;
    var c2s: [128]f32 = undefined;
    for (0..128) |j| {
        var gi: f32 = weights.lstm_b_ih[j];
        var gf: f32 = weights.lstm_b_ih[128 + j];
        var gg: f32 = weights.lstm_b_ih[256 + j];
        var go: f32 = weights.lstm_b_ih[384 + j];
        for (0..128) |i| {
            gi += weights.lstm_w_ih[j * 128 + i] * c4[0][i];
            gf += weights.lstm_w_ih[(128 + j) * 128 + i] * c4[0][i];
            gg += weights.lstm_w_ih[(256 + j) * 128 + i] * c4[0][i];
            go += weights.lstm_w_ih[(384 + j) * 128 + i] * c4[0][i];
            gi += weights.lstm_w_hh[j * 128 + i] * h[i];
            gf += weights.lstm_w_hh[(128 + j) * 128 + i] * h[i];
            gg += weights.lstm_w_hh[(256 + j) * 128 + i] * h[i];
            go += weights.lstm_w_hh[(384 + j) * 128 + i] * h[i];
        }
        gi += weights.lstm_b_hh[j];
        gf += weights.lstm_b_hh[128 + j];
        gg += weights.lstm_b_hh[256 + j];
        go += weights.lstm_b_hh[384 + j];
        const si = 1 / (1 + @exp(-gi));
        const sf = 1 / (1 + @exp(-gf));
        const sg = std.math.tanh(gg);
        const so = 1 / (1 + @exp(-go));
        c2s[j] = sf * c[j] + si * sg;
        h2[j] = so * std.math.tanh(c2s[j]);
    }

    var z = weights.final_b;
    for (0..128) |i| {
        z += weights.final_w[i] * @max(h2[i], 0);
    }
    return 1 / (1 + @exp(-z));
}

test "naive scalar pipeline matches optimized pipeline" {
    var prng = std.Random.DefaultPrng.init(0x777);
    var frame: [frame_size]f32 = undefined;

    var context: [64]f32 = @splat(0);
    var h: [128]f32 = @splat(0);
    var c: [128]f32 = @splat(0);
    var vad: VAD = .{};

    for (0..5) |_| {
        for (&frame) |*s| s.* = prng.random().float(f32) * 2 - 1;
        const fast = vad.process(&frame);
        const slow = naiveProcess(&context, &frame, &h, &c);
        try std.testing.expectApproxEqAbs(slow, fast, 1e-4);
        context = frame[448..512].*;
        h = vad.h;
        c = vad.c;
    }
}

//! Zilero VAD CLI.
//!
//! Reads 16-bit signed 16 kHz mono PCM from stdin (raw, or a .wav stream with
//! `-w`) and writes to stdout:
//!   simple mode  — one line per 32 ms frame: the probability, two decimals
//!   segment mode — JSONL voice segments when `-s <min silence ms>` is given
//!
//! Exit codes: 0 success, 1 IO/format error (message on stderr), 2 usage error.

const std = @import("std");
const zilero = @import("zilero");

const Config = struct {
    wav: bool = false,
    threshold: f32 = 0.55,
    min_silence_ms: ?u32 = null,
};

const RunError = error{
    BadWavHeader,
    BadFmtChunk,
    MissingFmtChunk,
    IoError,
};

const usage_text =
    \\simple voice activity detector cli. The cli read 16 bit signed 16Khz PCM samples from stdin and detect voice presence writing to stdout
    \\options:
    \\  -w  consider the input data as .wav file. (e.g.   cat file.wav | zilero-cli -w )
    \\  -p  <prob 0-1> vad probability threshold (defaults: 0.55)
    \\  -s  <min silence ms> detect segements of voice that have at least min silence between then. must be >= 100ms.
    \\output:
    \\   simple mode: for each frame received will output the vad probability with two decimal positions (e.g. 1.00, 0.86, 0.12) to stdout.
    \\   segment mode: if -s is provided will output voice segments jsonl with the follwing format: {"start":start_ms,"end":end_ms,"avg_prob":float}
;

pub fn main(init: std.process.Init) void {
    const io = init.io;
    var err_buf: [256]u8 = undefined;
    var err_w = std.Io.File.writerStreaming(std.Io.File.stderr(), io, &err_buf);

    const cfg = parseArgs(io, init.minimal.args, init.arena.allocator());
    run(io, cfg) catch |err| {
        const msg = switch (err) {
            error.BadWavHeader => "invalid wav: bad or truncated header",
            error.BadFmtChunk => "invalid wav: fmt chunk is not 16-bit mono PCM at 16000 Hz",
            error.MissingFmtChunk => "invalid wav: data chunk before fmt chunk",
            error.IoError => "io error",
        };
        std.Io.Writer.print(&err_w.interface, "{s}\n", .{msg}) catch {};
        std.Io.Writer.flush(&err_w.interface) catch {};
        std.process.exit(1);
    };
}

/// Prints the usage text and exits 2. Used for `-h` and for invalid args.
fn usageExit(w: *std.Io.Writer) noreturn {
    std.Io.Writer.writeAll(w, usage_text) catch {};
    std.Io.Writer.writeAll(w, "\n") catch {};
    std.Io.Writer.flush(w) catch {};
    std.process.exit(2);
}

/// Parses `-w`, `-p <f32>`, `-s <u32>`, `-h`. Prints the usage text and exits
/// 2 on `-h` or on any invalid argument.
fn parseArgs(io: std.Io, args: std.process.Args, alloc: std.mem.Allocator) Config {
    var out_buf: [4096]u8 = undefined;
    var out_w = std.Io.File.writerStreaming(std.Io.File.stdout(), io, &out_buf);
    var err_buf: [256]u8 = undefined;
    var err_w = std.Io.File.writerStreaming(std.Io.File.stderr(), io, &err_buf);

    var it = std.process.Args.Iterator.initAllocator(args, alloc) catch unreachable;
    defer it.deinit();
    _ = it.next(); // program name
    var cfg: Config = .{};
    while (true) {
        const arg = it.next() orelse break;
        if (std.mem.eql(u8, arg, "-h")) {
            usageExit(&out_w.interface);
        } else if (std.mem.eql(u8, arg, "-w")) {
            cfg.wav = true;
        } else if (std.mem.eql(u8, arg, "-p")) {
            const v = it.next() orelse usageExit(&err_w.interface);
            const p = std.fmt.parseFloat(f32, v) catch usageExit(&err_w.interface);
            if (!(p >= 0 and p <= 1)) usageExit(&err_w.interface);
            cfg.threshold = p;
        } else if (std.mem.eql(u8, arg, "-s")) {
            const v = it.next() orelse usageExit(&err_w.interface);
            const ms = std.fmt.parseInt(u32, v, 10) catch usageExit(&err_w.interface);
            if (ms < 100) usageExit(&err_w.interface);
            cfg.min_silence_ms = ms;
        } else {
            usageExit(&err_w.interface);
        }
    }
    return cfg;
}

fn run(io: std.Io, cfg: Config) RunError!void {
    var stdin_buf: [64 * 1024]u8 = undefined;
    var stdout_buf: [4096]u8 = undefined;
    var r = std.Io.File.reader(std.Io.File.stdin(), io, &stdin_buf);
    var w = std.Io.File.writerStreaming(std.Io.File.stdout(), io, &stdout_buf);

    if (cfg.wav) try skipWavHeader(&r);

    var vad: zilero.VAD = .{};
    var frame: [zilero.frame_size]i16 = undefined;
    var chunk: [1024]u8 = undefined;
    var frame_idx: u64 = 0;

    // Segment-mode state.
    var in_speech = false;
    var seg_start_ms: u64 = 0;
    var last_speech_end_ms: u64 = 0;
    var sum_prob: f64 = 0;
    var n_prob: u64 = 0;

    while (true) {
        const got = std.Io.Reader.readSliceShort(&r.interface, &chunk) catch return error.IoError;
        if (got == 0) break;
        const n_samples = got / 2;
        for (0..n_samples) |i| {
            const pair: [2]u8 = chunk[i * 2 ..][0..2].*;
            frame[i] = std.mem.readInt(i16, &pair, .little);
        }
        // A final partial frame is zero-padded to a full frame and processed.
        for (n_samples..zilero.frame_size) |i| frame[i] = 0;

        const prob = vad.process_i16(&frame);
        const start_ms = frame_idx * 32;
        const end_ms = start_ms + 32;
        frame_idx += 1;

        if (cfg.min_silence_ms) |min_silence| {
            if (prob >= cfg.threshold) {
                if (!in_speech) {
                    in_speech = true;
                    seg_start_ms = start_ms;
                    sum_prob = 0;
                    n_prob = 0;
                }
                last_speech_end_ms = end_ms;
                sum_prob += prob;
                n_prob += 1;
            } else if (in_speech and (end_ms - last_speech_end_ms) >= min_silence) {
                try emitSegment(&w.interface, seg_start_ms, last_speech_end_ms, sum_prob, n_prob);
                in_speech = false;
            }
        } else {
            std.Io.Writer.print(&w.interface, "{d:.2}\n", .{prob}) catch return error.IoError;
        }
    }

    if (in_speech) try emitSegment(&w.interface, seg_start_ms, last_speech_end_ms, sum_prob, n_prob);
    std.Io.Writer.flush(&w.interface) catch return error.IoError;
}

fn emitSegment(
    w: *std.Io.Writer,
    start_ms: u64,
    end_ms: u64,
    sum_prob: f64,
    n_prob: u64,
) RunError!void {
    const avg = sum_prob / @as(f64, @floatFromInt(n_prob));
    std.Io.Writer.print(
        w,
        "{{\"start\":{d},\"end\":{d},\"avg_prob\":{d:.3}}}\n",
        .{ start_ms, end_ms, avg },
    ) catch return error.IoError;
}

/// Skips a minimal RIFF/WAVE header: `RIFF <size> WAVE`, then chunks until
/// `data`. Requires a valid `fmt ` chunk (PCM, mono, 16 kHz, 16-bit) before
/// `data`; other chunks are skipped (+1 byte when their size is odd).
fn skipWavHeader(r: *std.Io.File.Reader) RunError!void {
    var hdr: [12]u8 = undefined;
    try readExact(r, &hdr);
    if (!std.mem.eql(u8, hdr[0..4], "RIFF") or !std.mem.eql(u8, hdr[8..12], "WAVE")) {
        return error.BadWavHeader;
    }

    var have_fmt = false;
    var tag: [8]u8 = undefined;
    while (true) {
        try readExact(r, &tag);
        const size_bytes: [4]u8 = tag[4..8].*;
        const size = std.mem.readInt(u32, &size_bytes, .little);

        if (std.mem.eql(u8, tag[0..4], "fmt ")) {
            if (size < 16) return error.BadFmtChunk;
            var fmt: [16]u8 = undefined;
            try readExact(r, &fmt);
            const af_bytes: [2]u8 = fmt[0..2].*;
            const ch_bytes: [2]u8 = fmt[2..4].*;
            const sr_bytes: [4]u8 = fmt[4..8].*;
            const bits_bytes: [2]u8 = fmt[14..16].*;
            const audio_format = std.mem.readInt(u16, &af_bytes, .little);
            const channels = std.mem.readInt(u16, &ch_bytes, .little);
            const sample_rate = std.mem.readInt(u32, &sr_bytes, .little);
            const bits = std.mem.readInt(u16, &bits_bytes, .little);
            if (audio_format != 1 or channels != 1 or sample_rate != 16000 or bits != 16) {
                return error.BadFmtChunk;
            }
            have_fmt = true;
            try skipBytes(r, size - 16);
        } else if (std.mem.eql(u8, tag[0..4], "data")) {
            if (!have_fmt) return error.MissingFmtChunk;
            return;
        } else {
            try skipBytes(r, size + @intFromBool(size % 2 == 1));
        }
    }
}

fn readExact(r: *std.Io.File.Reader, dst: []u8) RunError!void {
    var off: usize = 0;
    while (off < dst.len) {
        const n = std.Io.Reader.readSliceShort(&r.interface, dst[off..]) catch return error.IoError;
        if (n == 0) return error.BadWavHeader;
        off += n;
    }
}

fn skipBytes(r: *std.Io.File.Reader, n: usize) RunError!void {
    var buf: [4096]u8 = undefined;
    var left = n;
    while (left > 0) {
        const want = @min(left, buf.len);
        const got = std.Io.Reader.readSliceShort(&r.interface, buf[0..want]) catch return error.IoError;
        if (got == 0) return error.BadWavHeader;
        left -= got;
    }
}

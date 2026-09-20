const std = @import("std");
const weights = @import("weights");

pub const sample_rate = 16000;
pub const frame_size = 512;

comptime {
    _ = weights.final_b;
}

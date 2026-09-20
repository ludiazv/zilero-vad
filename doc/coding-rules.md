# Zig Coding Rules — Zilero VAD (skill for the coding worker)

Apply these rules to every file you write in this project. When a rule conflicts with something
you "remember" about Zig, the rule wins; when unsure about an API, read the installed std source.

## 1. Toolchain and verification loop
* Zig **0.16** only. Run `zig version` first; if it is not 0.16.x, stop and report.
* Never trust memory for `std` signatures. Find `lib_dir` with `zig env`, then `grep -rn "pub fn <name>" <lib_dir>/std/` and copy the real signature. Especially: `std.Io.Writer`, `std.Io.Reader`, `std.fs.File.stdout()/stdin()`, `std.process` args, `std.Build.*`, `std.json.parseFromSlice`.
* After every edit: `zig fmt .` → `zig build` → `zig build test`. Fix the first error, rerun. Do not batch up many changes before compiling.
* Commit after each green phase with a message `phase N: <what>`.

## 2. Project constraints (non-negotiable)
* Library (`src/zilero.zig`): **no allocations**, no `std.mem.Allocator` parameter anywhere, no heap. All buffers live inside the `VAD` struct or on the stack with comptime-known sizes.
* **No libc**, no `@cImport`, no `linkLibC`. Only `std` and builtins.
* Public surface of the library: `VAD` with `process`, `process_i16`, `reset`, plus `sample_rate` and `frame_size` constants. Everything else is non-`pub`.
* Weights come only from `@import("weights")` (generated at build time). Never hand-type a weight value.
* No global mutable state, no threads, no `comptime` tricks that make the code hard to read.

## 3. Naming and layout
* Types: `PascalCase` (`VAD` is the one accepted acronym exception). Functions: `camelCase` for private helpers; the three public functions keep the exact names given in the spec (`process`, `process_i16`, `reset`). Variables/fields/constants: `snake_case`.
* One concept per function; keep functions under ~40 lines. If a function needs a comment block to explain its sections, split it.
* Order inside a file: `const std = @import(...)`, other imports, public constants, public types, private helpers, then all `test` blocks at the bottom of the same file (Zig strips them from non-test builds; they may call private functions).
* `///` doc comments on every `pub` declaration; `//` comments only where the *why* is not obvious (e.g. `// PyTorch reflect pad: edge sample is not repeated`).

## 4. Types, slices, and numerics
* Prefer comptime-sized array pointers (`*const [256]f32`) over slices in hot paths; lengths in this model are all known at compile time — pass them as `comptime` parameters.
* Use explicit casts: `@intCast`, `@floatFromInt`, `@intFromFloat`, `@bitCast`, `@ptrCast`. Never rely on implicit widening you have not checked.
* `f32` everywhere in the model; no `f64` except when printing.
* Integers for sizes/indices: `usize`. For ms values in the CLI: `u64`.
* No `anytype` in public APIs. `anytype` is allowed in a private generic helper (e.g. the conv helper) only if it keeps the code shorter than a dedicated version.

## 5. SIMD rules
* Use whole-length vectors: `const V = @Vector(n, f32);` with `n` a comptime parameter equal to the array length. Never chunk by hand into 8/4 lanes — LLVM splits wide vectors into native-width instructions itself, and handles lengths like 129.
* Convert arrays to vectors with `const v: V = arr.*;` and back with `arr.* = v;`.
* Horizontal sum: `@reduce(.Add, v)`, always inside a function that has `@setFloatMode(.optimized)` (strict mode turns the reduction into a scalar chain).
* Elementwise: `@max(v, @as(V, @splat(0)))` for relu, `@exp(v)`, `@sqrt(v)` operate on vectors. Never mix scalars and vectors in one operator; `@splat` first.
* SIMD codegen comes from LLVM: use `ReleaseFast` for the CLI and benchmarks (x86_64 Debug uses the scalar self-hosted backend).
* No inline assembly, no target-specific intrinsics. Portability across x86_64/aarch64 comes from `@Vector` only.

## 6. Error handling
* Library functions **cannot fail** (they return `f32`/`void`, no error union). Invalid input is the caller's problem; the type system enforces frame length.
* Build tool and CLI: return `!void` from `main`, propagate with `try`. On user-facing errors print one line to stderr and exit with `1` (IO/format) or `2` (usage). No `catch unreachable` on IO.
* `unreachable` / `std.debug.assert` only for invariants provable from the code, never for input validation.
* No `std.debug.print` in library or CLI hot paths (fine in the build tool for the missing-tensor report).

## 7. IO (CLI and build tool)
* Use buffered readers/writers with caller-owned `[N]u8` buffers as 0.16 requires, and **flush** stdout before returning from `main`.
* Reading from stdin: loop until you have the bytes you need or hit EOF; a short read is normal.
* Endianness: parse binary with `std.mem.readInt(T, bytes[0..@sizeOf(T)], .little)`.

## 8. Tests
* Tests live in the file they test, after the code. `std.testing` only; start with `test { std.testing.refAllDecls(@This()); }`. Float comparisons via `expectApproxEqAbs`/`expectApproxEqRel`; exact equality only for determinism/reset tests.
* Deterministic inputs: `std.Random.DefaultPrng.init(fixed_seed)`. No wall clock, no external files except the optional `@embedFile` reference fixtures.
* Each test name says what it proves (`test "reset restores initial state"`). One assertion topic per test.

## 9. Keep it simple
* Do not add features not in the plan (no 8 kHz mode, no batching, no C ABI export, no options struct).
* Do not add a dependency, a build option, or an abstraction "for later".
* If a step in the plan is ambiguous, pick the simplest interpretation that compiles and passes tests, and note the choice in the commit message.

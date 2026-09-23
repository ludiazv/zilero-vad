"""
Pure-Python (stdlib only, no third-party deps) Conv1d matching the forward
semantics of torch.nn.Conv1d, minus dilation (fixed at 1), minus bias (this
op is always bias-free), minus groups (fixed at 1), and unbatched only
(single input, no batch dimension).

Equivalent to running, for a single sample:
    torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=padding,
                     dilation=1, groups=1, bias=False, padding_mode="zeros")

Shapes (matching nn.Conv1d with the batch dimension dropped):
    input  x:      (C_in, L_in)
    weight w:      (C_out, C_in, K)
    output:        (C_out, L_out)
        L_out = floor((L_in + 2*padding - K) / stride) + 1

Padding is symmetric zero-padding on both ends of the length dimension
(padding_mode="zeros", nn.Conv1d's default and only mode implemented here).
"""


def conv1d(
    x: list[list[float]],
    weight: list[list[list[float]]],
    stride: int = 1,
    padding: int = 0,
) -> list[list[float]]:
    """Functional Conv1d, no dilation, no bias, no groups, no batch dim.

    x: (C_in, L_in), weight: (C_out, C_in, K). Returns (C_out, L_out).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if padding < 0:
        raise ValueError("padding must be >= 0")

    c_in = len(x)
    l_in = len(x[0]) if c_in else 0
    c_out = len(weight)
    w_in = len(weight[0]) if c_out else 0
    k = len(weight[0][0]) if c_out else 0

    if w_in != c_in:
        raise ValueError(f"weight's in_channels ({w_in}) does not match input's ({c_in})")
    for row in x:
        if len(row) != l_in:
            raise ValueError("all input channels must have the same length")

    l_padded = l_in + 2 * padding
    if k > l_padded:
        raise ValueError(f"kernel size {k} is larger than padded input length {l_padded}")
    l_out = (l_padded - k) // stride + 1

    print(f"cin={c_in},lin={l_in},cout={c_out},k={k},w_in={w_in},pad={padding}")
    # Zero-pad each input channel once, up front, rather than bounds-checking
    # every tap (padding_mode="zeros", nn.Conv1d's default).
    px = [[0.0] * padding + list(row) + [0.0] * padding for row in x]

    out = [[0.0] * l_out for _ in range(c_out)]
    for o in range(c_out):
        w_o = weight[o]
        for t in range(l_out):
            base = t * stride
            acc = 0.0
            for i in range(c_in):
                row = px[i]
                w_row = w_o[i]
                for kk in range(k):
                    acc += w_row[kk] * row[base + kk]
            out[o][t] = acc
    return out


def conv1d_inv(
    x: list[list[float]],
    weight: list[list[list[float]]],
    stride: int = 1,
    padding: int = 0,
) -> list[list[float]]:
    """Same convolution as conv1d, computed from transposed inputs: x is
    (L_in, C_in) instead of (C_in, L_in), and weight is (C_out, K, C_in)
    instead of (C_out, C_in, K) -- kernel-tap-major, matching
    src/zilero.zig's conv3 weight layout ([out][k][in]). Padding is applied
    by skipping out-of-range taps rather than building a padded buffer
    (equivalent to zero-padding), also matching conv3. Output shape is
    still (C_out, L_out), same as conv1d.

    x: (L_in, C_in), weight: (C_out, K, C_in). Returns (C_out, L_out).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if padding < 0:
        raise ValueError("padding must be >= 0")

    l_in = len(x)
    c_in = len(x[0]) if l_in else 0
    c_out = len(weight)
    k = len(weight[0]) if c_out else 0
    w_in = len(weight[0][0]) if c_out and k else 0

    if w_in != c_in:
        raise ValueError(f"weight's in_channels ({w_in}) does not match input's ({c_in})")
    for row in x:
        if len(row) != c_in:
            raise ValueError("all input time-steps must have the same channel count")

    l_padded = l_in + 2 * padding
    if k > l_padded:
        raise ValueError(f"kernel size {k} is larger than padded input length {l_padded}")
    l_out = (l_padded - k) // stride + 1

    print(f"cin={c_in},lin={l_in},cout={c_out},k={k},w_in={w_in},pad={padding}")
    out = [[0.0] * l_out for _ in range(c_out)]
    for o in range(c_out):
        w_o = weight[o]
        for t in range(l_out):
            acc = 0.0
            for kk in range(k):
                r = t * stride + kk - padding
                if 0 <= r < l_in:
                    row = x[r]
                    wr = w_o[kk]
                    for i in range(c_in):
                        acc += row[i] * wr[i]
            out[o][t] = acc
    return out


if __name__ == "__main__":
    # Self-tests: no deps means no torch to check against, so these are
    # hand-computed against nn.Conv1d's documented semantics.

    # 1) kernel=3, stride=1, padding=1, single in/out channel, weights=1:
    #    equivalent to a length-3 zero-padded moving sum.
    x = [[1.0, 2.0, 3.0]]
    w = [[[1.0, 1.0, 1.0]]]
    out = conv1d(x, w, stride=1, padding=1)
    assert out == [[3.0, 6.0, 5.0]], out

    # 2) stride=2, padding=0, kernel=2, 2 in -> 1 out channel.
    x = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
    w = [[[1.0, 0.0], [0.0, 1.0]]]
    out = conv1d(x, w, stride=2, padding=0)
    # t=0: base=0 -> ch0[0]*1 + ch1[1]*1 = 1 + 20 = 21
    # t=1: base=2 -> ch0[2]*1 + ch1[3]*1 = 3 + 40 = 43
    assert out == [[21.0, 43.0]], out

    print("all conv1d self-tests passed")

    print("Directa:")
    x = [[2.0, 4.0, 3.0], [1.0, 7.0, 1]]
    w = [[[3.0, 2.0, 2.0], [5.0, 1.0, 3.0]]]
    out = conv1d(x, w, stride=1, padding=2)
    print(out)

    print("Inversa:")
    x = [[2.0, 1.0], [4.0, 7.0], [3.0, 1.0]]
    w = [[[3.0, 5.0], [2.0, 1.0], [2.0, 3.0]]]
    out = conv1d_inv(x, w, stride=1, padding=2)
    print(out)

    assert conv1d([[2.0, 4.0, 3.0], [1.0, 7.0, 1]], [[[3.0, 2.0, 2.0], [5.0, 1.0, 3.0]]], stride=1, padding=1) == conv1d_inv(
        [[2.0, 1.0], [4.0, 7.0], [3.0, 1.0]], [[[3.0, 5.0], [2.0, 1.0], [2.0, 3.0]]], stride=1, padding=1
    )
    print("conv1d_inv matches conv1d on transposed input/weight")

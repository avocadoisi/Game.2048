#!/usr/bin/env python3
"""Widen a value net so two nets of different capacity can be compared by `pairw`.

`pairw` runs both nets in one process, so both must have the same `h1` / `h2` (the widths are a
compile-time build option, `zig build nn -Dh1=… -Dh2=…`). A narrow net widens exactly: the extra
hidden units get zero weights, so their activation is `relu(0) = 0` and they contribute nothing —
the widened file computes the same value as the original, bit for bit.

    scripts/widen-net.py ai/solver/nn_weights.bin /tmp/ship128.bin --h1 128 --h2 128
    zig build nn -Doptimize=ReleaseFast -Dh1=128 -Dh2=128 -- 0 20 9000 0.003 1 pairw 30 /tmp/ship128.bin 1 - /tmp/cand.bin

Layout (little endian, `ai/solver/nn.zig`): 16-byte header (8-byte magic `2048-NN`, u32 version,
u32 endian marker 0x01020304), then f64 weights in struct order:
`w1[256][h1]`, `b1[h1]`, `w2[h2][h1]`, `b2[h2]`, `w3[h2]`, `b3`.
"""
import argparse
import struct
import sys

INPUTS = 256  # BoardCellCount * levels, see nn.zig


def dims(count: int):
    """Return (h1, h2) for a weight count, or None when no small width fits."""
    for h1 in range(8, 1025, 8):
        rest = count - INPUTS * h1 - h1 - 1
        if rest < 0:
            break
        # rest = h2*h1 + h2 + h2
        if rest % (h1 + 2) == 0:
            h2 = rest // (h1 + 2)
            if h2 > 0:
                return h1, h2
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--h1", type=int, required=True)
    ap.add_argument("--h2", type=int, required=True)
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    if len(raw) < 16 or raw[:8] != b"2048-NN\0":
        sys.exit(f"{a.src}: not a value net (bad magic)")
    version, marker = struct.unpack("<II", raw[8:16])
    body = raw[16:]
    values = list(struct.unpack("<%dd" % (len(body) // 8), body))
    found = dims(len(values))
    if found is None:
        sys.exit(f"{a.src}: {len(values)} values is not a net width")
    h1o, h2o = found
    h1n, h2n = a.h1, a.h2
    if h1n < h1o or h2n < h2o:
        sys.exit(f"cannot narrow {h1o}/{h2o} to {h1n}/{h2n}")

    i = 0
    w1 = [values[i + r * h1o : i + (r + 1) * h1o] for r in range(INPUTS)]
    i += INPUTS * h1o
    b1 = values[i : i + h1o]
    i += h1o
    w2 = [values[i + j * h1o : i + (j + 1) * h1o] for j in range(h2o)]
    i += h2o * h1o
    b2 = values[i : i + h2o]
    i += h2o
    w3 = values[i : i + h2o]
    i += h2o
    b3 = values[i]
    i += 1
    assert i == len(values)

    out = []
    for row in w1:
        out += row + [0.0] * (h1n - h1o)
    out += b1 + [0.0] * (h1n - h1o)
    for j in range(h2n):
        out += (w2[j] + [0.0] * (h1n - h1o)) if j < h2o else [0.0] * h1n
    out += b2 + [0.0] * (h2n - h2o)
    out += w3 + [0.0] * (h2n - h2o)
    out += [b3]

    with open(a.dst, "wb") as f:
        f.write(b"2048-NN\0" + struct.pack("<II", version, marker))
        f.write(struct.pack("<%dd" % len(out), *out))
    print(f"{a.src} ({h1o}/{h2o}) -> {a.dst} ({h1n}/{h2n}), {len(out)} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())

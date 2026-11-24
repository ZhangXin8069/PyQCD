"""Inspect a single explicit einsum using operand shapes, without allocating tensors."""

import argparse
from math import prod

import opt_einsum as oe
from opt_einsum.parser import parse_einsum_input


def parse_shape(value):
    if value == "()":
        return ()
    try:
        shape = tuple(int(axis) for axis in value.lower().split("x"))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}, expected e.g. 4x7 or '()'") from err
    if min(shape) < 1:
        raise argparse.ArgumentTypeError("use positive axis lengths")
    return shape


def axis_sizes(terms, shapes):
    """Return the local broadcast extents and the labels requiring broadcasting."""
    sizes = {}
    broadcast = set()
    for term, shape in zip(terms, shapes, strict=True):
        local = {}
        for axis, extent in zip(term, shape, strict=True):
            if axis in local and local[axis] != extent:
                raise ValueError(f"diagonal {axis!r} in {term!r} has unequal extents in {shape}")
            local[axis] = extent
            previous = sizes.get(axis, extent)
            if previous != extent:
                if min(previous, extent) != 1:
                    raise ValueError(f"incompatible extents for {axis!r}: {previous} and {extent}")
                broadcast.add(axis)
            sizes[axis] = max(previous, extent)
    return sizes, broadcast


def describe_step(terms, output, sizes, broadcast):
    if any(len(set(term)) < len(term) for term in terms):
        return "diagonal/trace"
    if broadcast:
        return "broadcast=" + "".join(sorted(broadcast))
    if len(terms) != 2:
        return "unary reduction/reorder" if len(terms) == 1 else "multi-operand contraction"

    a, b = terms
    single_sum = [k for k in sizes if k not in output and (k not in a or k not in b)]
    if single_sum:
        return "one-sided reduction=" + "".join(single_sum)
    batch = "".join(k for k in output if k in a and k in b)
    rows = "".join(k for k in output if k in a and k not in b)
    cols = "".join(k for k in output if k in b and k not in a)
    inner = "".join(k for k in a if k in b and k not in output)
    if not inner:
        return "elementwise/outer product"

    groups = (batch, rows, cols, inner)
    dims = tuple(prod(sizes[k] for k in group) for group in groups)
    labels = ",".join(group or "1" for group in groups)
    return f"B,M,N,K=[{labels}] {dims}  dense_complex_MACs={prod(dims)}"


def report(equation, shapes, optimizer, itemsize):
    path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=optimizer)
    live = [(shape, False) for shape in shapes]
    input_elems = sum(prod(s) for s in shapes)
    peak_temps = 0
    print(f"\n{optimizer}  path={path}  oe_cost={info.opt_cost}")
    for number, (positions, _, expression, _, _) in enumerate(info.contraction_list, 1):
        lhs, out = expression.split("->")
        terms = lhs.split(",")
        # opt_einsum orders positions and the two terms together, right to left in 'live'.
        held = sum(prod(shape) for shape, intermediate in live if intermediate)
        operands = [live.pop(pos)[0] for pos in positions]
        sizes, broadcast = axis_sizes(terms, operands)
        shape = tuple(sizes[k] for k in out)
        elements = prod(shape)
        peak_temps = max(peak_temps, held + elements)
        detail = describe_step(terms, out, sizes, broadcast)
        print(f"{number:3d}  {expression}  shape={shape}  bytes={elements * itemsize}  {detail}")
        live.append((shape, True))
    print(f"input_logical_bytes={input_elems * itemsize}  "
          f"model_peak_bytes={(input_elems + peak_temps) * itemsize}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("equation", help="einsum with explicit -> output (empty for scalar)")
    parser.add_argument("shapes", nargs="+", type=parse_shape, help="e.g. 4x4x7 or '()'")
    parser.add_argument("--optimizers", nargs="+", choices=("greedy", "auto", "dp", "optimal"),
                        default=["greedy"])
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    args = parser.parse_args()
    if args.equation.count("->") != 1:
        parser.error("an explicit -> output is required")
    itemsize = {"complex64": 8, "complex128": 16}[args.dtype]
    try:
        inputs, _, _ = parse_einsum_input((args.equation, *args.shapes), shapes=True)
        axis_sizes(inputs.split(","), args.shapes)
        for optimizer in args.optimizers:
            report(args.equation, args.shapes, optimizer, itemsize)
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()

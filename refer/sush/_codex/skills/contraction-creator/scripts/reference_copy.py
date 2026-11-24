#!/usr/bin/env python3
"""Copy/check a reference plus declared, unique UTF-8 byte replacements.

This tool never imports a contraction program or infers physical parameters.
The skill's input contract must be resolved before it is used for generation.
"""

import argparse
import difflib
import hashlib
from pathlib import Path
import sys


def patched_bytes(reference, replacements):
    data = reference.read_bytes()
    for old, new in replacements:
        old_bytes, new_bytes = old.encode("utf-8"), new.encode("utf-8")
        if not old_bytes or old_bytes == new_bytes:
            raise ValueError("replacement must have nonempty, different old/new text")
        count = data.count(old_bytes)
        if count != 1:
            raise ValueError(f"replacement must match exactly once; found {count}: {old!r}")
        data = data.replace(old_bytes, new_bytes, 1)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("copy", "check"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replace", nargs=2, action="append", default=[],
                        metavar=("OLD", "NEW"),
                        help="unique literal replacement, in order; repeat for each edit")
    args = parser.parse_args()
    try:
        if args.reference.resolve() == args.output.resolve():
            raise ValueError("reference and output must be different files")
        expected = patched_bytes(args.reference, args.replace)
        if args.action == "copy":
            # Exclusive creation also protects an existing output from partial patches.
            with args.output.open("xb") as stream:
                stream.write(expected)
        actual = args.output.read_bytes()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if actual != expected:
        print("FAIL: output differs from the reference plus declared replacements", file=sys.stderr)
        before = expected.decode("utf-8", errors="replace").splitlines(keepends=True)
        after = actual.decode("utf-8", errors="replace").splitlines(keepends=True)
        sys.stderr.writelines(difflib.unified_diff(
            before, after, fromfile="expected", tofile=str(args.output)))
        return 1

    mode = "exact reference bytes" if not args.replace else "reference plus declared replacements"
    print(f"PASS: {mode}; sha256={hashlib.sha256(actual).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Printable information about a contraction path."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence

__all__ = ["PathInfo"]


class PathInfo:
    """A printable object to contain information about a contraction path."""

    def __init__(
        self,
        contraction_list: Iterable[Any],
        input_subscripts: str,
        output_subscript: str,
        indices: frozenset[str],
        path: Sequence[Sequence[int]],
        scale_list: Sequence[int],
        naive_cost: int,
        opt_cost: int,
        size_list: Sequence[int],
        size_dict: dict[str, int],
    ) -> None:
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.path = path
        self.indices = indices
        self.scale_list = scale_list
        self.naive_cost = Decimal(naive_cost)
        self.opt_cost = Decimal(opt_cost)
        self.speedup = self.naive_cost / max(self.opt_cost, Decimal(1))
        self.size_list = size_list
        self.size_dict = size_dict

        self.shapes = [tuple(size_dict[k] for k in ks) for ks in input_subscripts.split(",")]
        self.eq = f"{input_subscripts}->{output_subscript}"
        self.largest_intermediate = Decimal(max(size_list, default=1))

    def __repr__(self) -> str:
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = [
            f"  Complete contraction:  {self.eq}\n",
            f"         Naive scaling:  {len(self.indices)}\n",
            f"     Optimized scaling:  {max(self.scale_list, default=0)}\n",
            f"      Naive FLOP count:  {self.naive_cost:.3e}\n",
            f"  Optimized FLOP count:  {self.opt_cost:.3e}\n",
            f"   Theoretical speedup:  {self.speedup:.3e}\n",
            f"  Largest intermediate:  {self.largest_intermediate:.3e} elements\n",
            "-" * 80 + "\n",
            "{:>6} {:>11} {:>22} {:>37}\n".format(*header),
            "-" * 80,
        ]

        for n, contraction in enumerate(self.contraction_list):
            _, _, einsum_str, remaining, do_blas = contraction

            if remaining is not None:
                remaining_str = ",".join(remaining) + "->" + self.output_subscript
            else:
                remaining_str = "..."

            size_remaining = max(0, 56 - max(22, len(einsum_str)))
            path_run = (
                self.scale_list[n],
                do_blas,
                einsum_str,
                remaining_str,
                size_remaining,
            )
            path_print.append("\n{:>4} {:>14} {:>22}    {:>{}}".format(*path_run))

        return "".join(path_print)

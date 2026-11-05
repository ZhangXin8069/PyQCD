"""Pure Python contraction path planning: greedy, optimal, branch bound and dp."""

import bisect
import functools
import heapq
import itertools
import operator
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from .blas import can_blas
from .helpers import compute_size_by_dict, find_contraction, flop_count
from .parser import find_output_shape, parse_einsum_input

Step = tuple[tuple[int, ...], frozenset, str, tuple[str, ...] | None, str | bool]

_UNLIMITED_MEM = {-1, None, float("inf")}


class PathOptimizer:
    """Base class for path optimizers.

    Subclasses provide ``__call__(inputs, output, size_dict, memory_limit)`` and return
    a list of int tuples; the arguments are bound to the first call for statefulness.
    """

    def _check_args_against_first_call(self, inputs, output, size_dict) -> None:
        args = (inputs, output, size_dict)
        if not hasattr(self, "_first_call_args"):
            self._first_call_args = args
        elif args != self._first_call_args:
            raise ValueError(
                "The arguments specifying the contraction that this path optimizer "
                "instance was called with have changed - try creating a new instance."
            )

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        raise NotImplementedError


def ssa_to_linear(ssa_path):
    """Convert a static single assignment path to recycled linear ids."""
    n = sum(map(len, ssa_path)) - len(ssa_path) + 1
    ids = list(range(n))
    path = []
    ssa = n
    for scon in ssa_path:
        con = sorted([bisect.bisect_left(ids, s) for s in scon])
        for j in reversed(con):
            ids.pop(j)
        ids.append(ssa)
        path.append(con)
        ssa += 1
    return [tuple(x) for x in path]


def linear_to_ssa(path):
    """Convert a path with recycled linear ids to static single assignment ids."""
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def calc_k12_flops(inputs, output, remaining, i, j, size_dict):
    """Indices and flop cost of contracting ``i`` and ``j`` with ``remaining`` still live."""
    k1, k2 = inputs[i], inputs[j]
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))

    k12 = either & keep
    cost = flop_count(either, bool(shared - keep), 2, size_dict)

    return k12, cost


def _compute_oversize_flops(inputs, remaining, output, size_dict) -> int:
    """Flop cost of fusing every remaining tensor at once, used past a memory limit."""
    idx_contraction = frozenset.union(*map(inputs.__getitem__, remaining))
    inner = idx_contraction - output
    num_terms = len(remaining)
    return flop_count(idx_contraction, bool(inner), num_terms, size_dict)


def optimal(inputs, output, size_dict, memory_limit=None):
    """Exhaustive depth-first search over pairwise contractions, sieved by cost."""
    inputs_set = tuple(map(frozenset, inputs))
    output_set = frozenset(output)

    best_flops = {"flops": float("inf")}
    best_ssa_path = {"ssa_path": (tuple(range(len(inputs))),)}
    size_cache: dict = {}
    result_cache: dict = {}

    def _optimal_iterate(path, remaining, inputs, flops):
        if len(remaining) == 1:
            best_flops["flops"] = flops
            best_ssa_path["ssa_path"] = path
            return

        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = calc_k12_flops(inputs, output_set, remaining, i, j, size_dict)

            new_flops = flops + flops12
            if new_flops >= best_flops["flops"]:
                continue

            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output_set, size_dict)
                    if new_flops < best_flops["flops"]:
                        best_flops["flops"] = new_flops
                        best_ssa_path["ssa_path"] = path + (tuple(remaining),)
                    continue

            _optimal_iterate(
                path=path + ((i, j),),
                inputs=inputs + (k12,),
                remaining=remaining - {i, j} | {len(inputs)},
                flops=new_flops,
            )

    _optimal_iterate(path=(), inputs=inputs_set, remaining=set(range(len(inputs))), flops=0)

    return ssa_to_linear(best_ssa_path["ssa_path"])


def better_flops_first(flops, size, best_flops, best_size) -> bool:
    """Compare two candidates on ``(flops, size)``."""
    return (flops, size) < (best_flops, best_size)


def better_size_first(flops, size, best_flops, best_size) -> bool:
    """Compare two candidates on ``(size, flops)``."""
    return (size, flops) < (best_size, best_flops)


_BETTER_FNS = {"flops": better_flops_first, "size": better_size_first}


def get_better_fn(key: str):
    """Comparison function for the ``minimize`` flag ``key``."""
    return _BETTER_FNS[key]


def cost_memory_removed(size12, size1, size2, k12, k1, k2):
    """Heuristic cost: total reduction in memory of a contraction."""
    return size12 - size1 - size2


def cost_memory_removed_jitter(size12, size1, size2, k12, k1, k2):
    """Jittered memory reduction, used to break ties randomly."""
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    "memory-removed": cost_memory_removed,
    "memory-removed-jitter": cost_memory_removed_jitter,
}


class BranchBound(PathOptimizer):
    """Depth-first branch and bound search over pairwise contractions."""

    def __init__(self, nbranch=None, cutoff_flops_factor=4, minimize="flops", cost_fn="memory-removed"):
        if (nbranch is not None) and nbranch < 1:
            raise ValueError(f"The number of branches must be at least one, `nbranch={nbranch}`.")

        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn = _COST_FNS.get(cost_fn, cost_fn)

        self.better = get_better_fn(minimize)
        self.best: dict = {"flops": float("inf"), "largest_intermediate": float("inf")}
        self.best_progress = defaultdict(lambda: float("inf"))

    @property
    def path(self):
        """Best path found so far."""
        return ssa_to_linear(self.best["ssa_path"])

    def __call__(self, inputs_, output_, size_dict, memory_limit=None):
        self._check_args_against_first_call(inputs_, output_, size_dict)

        inputs = tuple(map(frozenset, inputs_))
        output = frozenset(output_)

        output_size = compute_size_by_dict(output, size_dict)

        size_cache = {k: compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache: dict = {}

        def _branch_iterate(path, inputs, remaining, flops, size):
            if len(remaining) == 1:
                self.best["largest_intermediate"] = size
                self.best["flops"] = flops
                self.best["ssa_path"] = path
                return

            def _assess_candidate(k1, k2, i, j):
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)

                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                # Assess the all-terms fallback before pruning on a pairwise
                # intermediate that will never be formed.
                if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output_, size_dict)
                    new_size = max(size, output_size)
                    if self.better(new_flops, new_size, self.best["flops"], self.best["largest_intermediate"]):
                        self.best["flops"] = new_flops
                        self.best["largest_intermediate"] = new_size
                        self.best["ssa_path"] = path + (tuple(remaining),)
                    return None

                new_flops = flops + flops12
                new_size = max(size, size12)

                if not self.better(new_flops, new_size, self.best["flops"], self.best["largest_intermediate"]):
                    return None

                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
                    return None

                size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)

                return cost, flops12, new_flops, new_size, (i, j), k12

            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = j, i
                k1, k2 = inputs[i], inputs[j]

                if k1.isdisjoint(k2):
                    continue

                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = j, i
                    k1, k2 = inputs[i], inputs[j]
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)

            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(
                    path=path + ((i, j),),
                    inputs=inputs + (k12,),
                    remaining=(remaining - {i, j}) | {len(inputs)},
                    flops=new_flops,
                    size=new_size,
                )
                bi += 1

        _branch_iterate(path=(), inputs=inputs, remaining=set(range(len(inputs))), flops=0, size=0)

        return self.path


def branch(
    inputs,
    output,
    size_dict,
    memory_limit=None,
    nbranch=None,
    cutoff_flops_factor=4,
    minimize="flops",
    cost_fn="memory-removed",
):
    """Branch and bound path search, exploring ``nbranch`` candidates per step."""
    optimizer = BranchBound(
        nbranch=nbranch, cutoff_flops_factor=cutoff_flops_factor, minimize=minimize, cost_fn=cost_fn
    )
    return optimizer(inputs, output, size_dict, memory_limit)


branch_all = functools.partial(branch, nbranch=None)
branch_2 = functools.partial(branch, nbranch=2)
branch_1 = functools.partial(branch, nbranch=1)


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = cost_fn(
        compute_size_by_dict(k12, sizes),
        footprints[k1],
        footprints[k2],
        k12,
        k1,
        k2,
    )
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn):
    candidates = (_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn) for k2 in k2s)
    if push_all:
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(dim_to_keys, dim_ref_counts, dims) -> None:
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Take the minimum cost candidate, skipping candidates that went stale."""
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None
    return cost, k1, k2, k12


def ssa_greedy_optimize(inputs, output, sizes, choose_fn=None, cost_fn="memory-removed"):
    """Greedy path in static single assignment form: Hadamard, then contraction, then outer."""
    if len(inputs) == 1:
        return [(0,)]

    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        push_all = True

    fs_inputs = [frozenset(x) for x in inputs]
    output = frozenset(output) | frozenset.intersection(*fs_inputs)

    remaining: dict = {}
    ssa_ids = itertools.count(len(fs_inputs))
    ssa_path: list = []
    for ssa_id, key in enumerate(fs_inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    dim_ref_counts = {
        count: {dim for dim, keys in dim_to_keys.items() if len(keys) >= count} - output for count in [2, 3]
    }

    footprints = {key: compute_size_by_dict(key, sizes) for key in remaining}

    queue: list = []
    for dim, dim_keys in dim_to_keys.items():
        dim_keys_list = sorted(dim_keys, key=remaining.__getitem__)
        for i, k1 in enumerate(dim_keys_list[:-1]):
            k2s_guess = dim_keys_list[1 + i :]
            _push_candidate(
                output, sizes, remaining, footprints, dim_ref_counts, k1, k2s_guess, queue, push_all, cost_fn
            )

    while queue:
        con = choose_fn(queue, remaining)
        if con is None:
            continue
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | (k2 - output))
        footprints[k12] = compute_size_by_dict(k12, sizes)

        k1 = k12
        k2s = {k2 for dim in k1 for k2 in dim_to_keys[dim]}
        k2s.discard(k1)
        if k2s:
            _push_candidate(
                output, sizes, remaining, footprints, dim_ref_counts, k1, list(k2s), queue, push_all, cost_fn
            )

    final_queue = [(compute_size_by_dict(key & output, sizes), ssa_id, key) for key, ssa_id in remaining.items()]
    heapq.heapify(final_queue)
    _, ssa_id1, k1 = heapq.heappop(final_queue)
    while final_queue:
        _, ssa_id2, k2 = heapq.heappop(final_queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(final_queue, (cost, ssa_id12, k12))

    return ssa_path


def greedy(inputs, output, size_dict, memory_limit=None, choose_fn=None, cost_fn="memory-removed"):
    """Greedy path: Hadamard products, then size reducing contractions, then outer products."""
    if memory_limit not in _UNLIMITED_MEM:
        return branch(inputs, output, size_dict, memory_limit, nbranch=1, cost_fn=cost_fn)

    ssa_path = ssa_greedy_optimize(inputs, output, size_dict, cost_fn=cost_fn, choose_fn=choose_fn)
    return ssa_to_linear(ssa_path)


def _tree_to_sequence(tree):
    """Flatten a contraction tree into a linear path."""
    if type(tree) == int:
        return []

    c: list = [tree]
    t: list = []
    s: list = []

    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, ())

        for i in sorted([i for i in j if type(i) == int]):
            s[0] += (sum(1 for q in t if q < i),)
            t.insert(s[0][-1], i)

        for i_tup in [i_tup for i_tup in j if type(i_tup) != int]:
            s[0] += (len(t) + len(c),)
            c.append(i_tup)

    return s


def _find_disconnected_subgraphs(inputs, output):
    """Group input tensors that are linked by shared summation indices."""
    subgraphs = []
    unused_inputs = set(range(len(inputs)))

    i_sum = frozenset.union(*inputs) - output

    while len(unused_inputs) > 0:
        g = set()
        q = [unused_inputs.pop()]
        while len(q) > 0:
            j = q.pop()
            g.add(j)
            i_tmp = i_sum & inputs[j]
            n = {k for k in unused_inputs if len(i_tmp & inputs[k]) > 0}
            q.extend(n)
            unused_inputs.difference_update(n)

        subgraphs.append(g)

    return [frozenset(x) for x in subgraphs]


def _bitmap_select(s, seq):
    """Elements of ``seq`` whose bit is set in the bitmap ``s``."""
    return (x for x, b in zip(seq, bin(s)[:1:-1]) if b == "1")


def _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2):
    """Effective outer indices of the intermediate represented by subgraph ``s``."""
    r = g & (all_tensors ^ s)
    if r:
        i_r = frozenset.union(*_bitmap_select(r, inputs))
    else:
        i_r = frozenset()
    i_contract = i1_cut_i2_wo_output - i_r
    return i1_union_i2 - i_contract


def _dp_compare_flops(
    cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs,
    i1_cut_i2_wo_output, memory_limit, contract1, contract2,
):
    """DP merge step minimizing total flops."""
    cost = cost1 + cost2 + compute_size_by_dict(i1_union_i2, size_dict)
    if cost <= cost_cap:
        s = s1 | s2
        if s not in xn or cost < xn[s][1]:
            i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
            mem = compute_size_by_dict(i, size_dict)
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_size(
    cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs,
    i1_cut_i2_wo_output, memory_limit, contract1, contract2,
):
    """DP merge step minimizing the largest intermediate."""
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = max(cost1, cost2, mem)
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_write(
    cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs,
    i1_cut_i2_wo_output, memory_limit, contract1, contract2,
):
    """DP merge step minimizing the total size written."""
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = cost1 + cost2 + mem
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


DEFAULT_COMBO_FACTOR = 64


def _dp_compare_combo(
    cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs,
    i1_cut_i2_wo_output, memory_limit, contract1, contract2, factor=DEFAULT_COMBO_FACTOR, combine=sum,
):
    """DP merge step minimizing a combination of flops and written size."""
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    f = compute_size_by_dict(i1_union_i2, size_dict)
    cost = cost1 + cost2 + combine((f, factor * mem))
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


minimize_finder = re.compile(r"(flops|size|write|combo|limit)-*(\d*)")


@functools.lru_cache(128)
def _parse_minimize(minimize):
    """Local DP scoring function for ``minimize``, plus the naive cost scale."""
    if minimize == "flops":
        return _dp_compare_flops, 1
    elif minimize == "size":
        return _dp_compare_size, 1
    elif minimize == "write":
        return _dp_compare_write, 1
    elif callable(minimize):
        return minimize, float("inf")

    match = minimize_finder.fullmatch(minimize)
    if match is None:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")

    minimize, custom_factor = match.groups()
    factor = float(custom_factor) if custom_factor else DEFAULT_COMBO_FACTOR
    if minimize == "combo":
        return functools.partial(_dp_compare_combo, factor=factor, combine=sum), float("inf")
    elif minimize == "limit":
        return functools.partial(_dp_compare_combo, factor=factor, combine=max), float("inf")
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


def simple_tree_tuple(seq):
    """Left to right binary tree over ``seq``."""
    return functools.reduce(lambda x, y: (x, y), seq)


def _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts):
    """Split off single term index summations before the DP search."""
    i_single = frozenset(i for i, c in enumerate(all_inds) if ind_counts[c] == 1)
    inputs_parsed: list = []
    inputs_done: list = []
    inputs_contractions: list = []
    for j, i in enumerate(inputs):
        i_reduced = i - i_single
        if (not i_reduced) and (len(i) > 0):
            inputs_done.append((j,))
        else:
            inputs_parsed.append(i_reduced)
            inputs_contractions.append((j,) if i_reduced != i else j)

    return inputs_parsed, inputs_done, inputs_contractions


class DynamicProgramming(PathOptimizer):
    """Dynamic programming over tensor subsets, no intermediate outer products."""

    def __init__(self, minimize="flops", cost_cap=True, search_outer=False) -> None:
        self.minimize = minimize
        self.search_outer = search_outer
        self.cost_cap = cost_cap

    def __call__(self, inputs_, output_, size_dict_, memory_limit_=None):
        _check_contraction, naive_scale = _parse_minimize(self.minimize)
        _check_outer = (lambda x: True) if self.search_outer else (lambda x: x)

        ind_counts = Counter(itertools.chain(*inputs_, output_))
        all_inds = tuple(ind_counts)

        symbol2int = {c: j for j, c in enumerate(all_inds)}
        inputs = [frozenset(symbol2int[c] for c in i) for i in inputs_]
        output = frozenset(symbol2int[c] for c in output_)
        size_dict_canonical = {symbol2int[c]: v for c, v in size_dict_.items() if c in symbol2int}
        size_dict = [size_dict_canonical[j] for j in range(len(size_dict_canonical))]
        # A zero sized axis may be reduced away or live in another subgraph, so it
        # must not drive the search bound for every subgraph.
        naive_cost = naive_scale * len(inputs) * functools.reduce(operator.mul, (max(1, d) for d in size_dict), 1)

        inputs, inputs_done, inputs_contractions = _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts)

        if not inputs:
            return _tree_to_sequence(simple_tree_tuple(inputs_done))

        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1] * len(inputs_done)

        if self.search_outer:
            subgraphs = [frozenset(range(len(inputs)))]
        else:
            subgraphs = _find_disconnected_subgraphs(inputs, output)

        all_tensors = (1 << len(inputs)) - 1

        for g in subgraphs:
            x: list = [None] * 2 + [{} for j in range(len(g) - 1)]
            x[1] = {1 << j: (inputs[j], 0, inputs_contractions[j]) for j in g}

            bitmap_g = functools.reduce(lambda x, y: x | y, (1 << j for j in g))

            subgraph_inds = frozenset.union(*_bitmap_select(bitmap_g, inputs))
            if self.cost_cap is True:
                cost_cap = compute_size_by_dict(subgraph_inds & output, size_dict)
            elif self.cost_cap is False:
                cost_cap = float("inf")
            else:
                cost_cap = self.cost_cap
            if len(subgraph_inds) == 0:
                cost_increment = 2
            else:
                cost_increment = max(min(map(size_dict.__getitem__, subgraph_inds)), 2)

            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]

                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, contract1) in x[m].items():
                            for s2, (i2, cost2, contract2) in x[n - m].items():
                                if (not s1 & s2) and (m != n - m or s1 < s2):
                                    i1_cut_i2_wo_output = (i1 & i2) - output

                                    if _check_outer(i1_cut_i2_wo_output):
                                        i1_union_i2 = i1 | i2
                                        _check_contraction(
                                            cost1,
                                            cost2,
                                            i1_union_i2,
                                            size_dict,
                                            cost_cap,
                                            s1,
                                            s2,
                                            xn,
                                            bitmap_g,
                                            all_tensors,
                                            inputs,
                                            i1_cut_i2_wo_output,
                                            memory_limit_,
                                            contract1,
                                            contract2,
                                        )

                if (cost_cap > naive_cost) and (len(x[-1]) == 0):
                    raise RuntimeError("No contraction found for given `memory_limit`.")

                cost_cap = cost_increment * cost_cap

            i, cost, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(compute_size_by_dict(i, size_dict))

        # Ascending subgraph size keeps the cheap contractions first; outer
        # products stay pairwise so they can still use BLAS.
        subgraph_contractions = [
            subgraph_contractions[j]
            for j in sorted(
                range(len(subgraph_contractions_size)),
                key=subgraph_contractions_size.__getitem__,
            )
        ]

        tree = simple_tree_tuple(subgraph_contractions)
        return _tree_to_sequence(tree)


def dynamic_programming(inputs, output, size_dict, memory_limit=None, **kwargs):
    """Dynamic programming path search; ``kwargs`` go to :class:`DynamicProgramming`."""
    optimizer = DynamicProgramming(**kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)


_AUTO_CHOICES: dict = {}
for i in range(1, 5):
    _AUTO_CHOICES[i] = optimal
for i in range(5, 7):
    _AUTO_CHOICES[i] = branch_all
for i in range(7, 9):
    _AUTO_CHOICES[i] = branch_2
for i in range(9, 15):
    _AUTO_CHOICES[i] = branch_1


def auto(inputs, output, size_dict, memory_limit=None):
    """Pick an optimizer from the number of input tensors (``greedy`` from 15 on)."""
    return _AUTO_CHOICES.get(len(inputs), greedy)(inputs, output, size_dict, memory_limit)


_PATH_FNS: dict = {
    "auto": auto,
    "optimal": optimal,
    "branch-all": branch_all,
    "branch-2": branch_2,
    "branch-1": branch_1,
    "greedy": greedy,
    "eager": greedy,
    "opportunistic": greedy,
    "dp": dynamic_programming,
    "dynamic-programming": dynamic_programming,
}


def _get_path_fn(path_type: str):
    """Built-in path search function for ``path_type``."""
    path_type = path_type.lower()
    if path_type not in _PATH_FNS:
        raise KeyError(f"Path optimizer '{path_type}' not found, valid options are {set(_PATH_FNS.keys())}.")
    return _PATH_FNS[path_type]


def _choose_memory_arg(memory_limit, size_list):
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError("memory_limit must be None, int, or the string Literal['max_input'].")

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


def _parse_equation(eq: str, shapes: Sequence[Sequence[int]]):
    """Split ``eq`` into input terms, term shapes, output string and index sizes."""
    input_subscripts, output_subscript, _ = parse_einsum_input((eq, *shapes), shapes=True)
    input_list = input_subscripts.split(",")
    input_shapes = [tuple(shape) for shape in shapes]

    size_dict: dict = {}
    for tnum, term in enumerate(input_list):
        shape = input_shapes[tnum]
        if len(shape) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(shape[cnum])
            if char in size_dict:
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    return input_list, input_shapes, output_subscript, size_dict


def optimize(
    eq: str,
    shapes: Sequence[Sequence[int]],
    optimize="auto",
    memory_limit: int | None = None,
    use_blas: bool = True,
) -> list[Step]:
    """Plan the contraction of ``eq`` with the given ``shapes``.

    Returns the contraction list as ``(inds, idx_removed, einsum_str, remaining, do_blas)``
    steps, ``inds`` being the consuming operand positions in descending order.
    """
    input_list, input_shapes, output_subscript, size_dict = _parse_equation(eq, shapes)
    input_sets = [frozenset(x) for x in input_list]
    output_set = frozenset(output_subscript)

    size_list = [compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    if optimize is True or optimize is None:
        optimize = "auto"

    num_ops = len(input_list)
    if optimize is False:
        path: list = [tuple(range(num_ops))]
    elif not isinstance(optimize, (str, PathOptimizer)):
        path = [tuple(p) for p in optimize]
    elif num_ops <= 2:
        path = [tuple(range(num_ops))]
    elif isinstance(optimize, PathOptimizer):
        path = optimize(input_sets, output_set, size_dict, memory_arg)
    else:
        path = _get_path_fn(optimize)(input_sets, output_set, size_dict, memory_arg)

    contraction_list: list[Step] = []
    num_steps = len(path)
    for cnum, contract_inds in enumerate(path):
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        out_inds, input_sets, idx_removed = find_contraction(contract_inds, input_sets, output_set)

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        do_blas = can_blas(tmp_inputs, "".join(out_inds), idx_removed, tmp_shapes) if use_blas else False

        if (cnum - num_steps) == -1:
            idx_result = output_subscript
        else:
            # Input order of the tmp tensors minimizes the transpositions.
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shapes.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        remaining = tuple(input_list) if len(input_list) <= 20 else None

        contraction_list.append((contract_inds, idx_removed, einsum_str, remaining, do_blas))

    return contraction_list

#!/usr/bin/env python3
"""`paths` 单函数对照：register_path_fn / get_path_fn / 各规划器代价一致。

`torch.einsum` 不做路径规划，因此对照对象是 **opt_einsum**；
规划结果的可执行性与数值正确性由 check_contract_path.py 用 torch 校验。
"""

from __future__ import annotations

import argparse

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple import _common


OPTIMIZERS = ('greedy', 'dp', 'optimal', 'branch-2')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('paths')
    target = _common.target_module()
    oracle = _common.oracle_module()

    # 1) 内置规划器：同一表达式下 opt_cost 必须一致
    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        equation, operands = _common.random_case(rng, max_operands=4)
        for optimizer in OPTIMIZERS:
            try:
                target_path, target_info = target.contract_path(
                    equation, *operands, optimize=optimizer)
                oracle_path, oracle_info = oracle.contract_path(
                    equation, *operands, optimize=optimizer)
            except Exception as exc:  # noqa: BLE001 - 某侧不支持则记录
                tally.skip(f'case{index}/{optimizer}',
                           f'{type(exc).__name__}: {exc}')
                continue
            tally.check(int(target_info.opt_cost) == int(oracle_info.opt_cost),
                        f'case{index}/{optimizer}-cost',
                        f'{target_info.opt_cost} != {oracle_info.opt_cost}')
            tally.check(len(target_path) == len(oracle_path),
                        f'case{index}/{optimizer}-step-count',
                        f'{len(target_path)} != {len(oracle_path)}')

    # 2) register_path_fn / get_path_fn：注册一个自定义规划器并按其名字调用
    def custom_path(inputs, output, size_dict, memory_limit=None):
        # 从尾部两两收缩：收缩 (n-2, n-1) 后新张量占据 n-2 位置，
        # 因此下一轮仍是 (len-2, len-1)，直到只剩一个操作数。
        remaining = list(range(len(inputs)))
        steps = []
        while len(remaining) > 1:
            left, right = len(remaining) - 2, len(remaining) - 1
            steps.append((left, right))
            remaining = remaining[:-2] + [left]
        return steps

    for module, label in ((target, 'pyqcd'), (oracle, 'opt')):
        try:
            module.paths.register_path_fn('simple-custom', custom_path)
            resolved = module.paths.get_path_fn('simple-custom')
        except Exception as exc:  # noqa: BLE001 - 注册失败只记录
            tally.check(False, f'{label}/register-custom',
                        f'{type(exc).__name__}: {exc}')
            continue
        tally.check(resolved is not None, f'{label}/register-custom', '未注册成功')

    rng = _common.seeded_rng(args.seed + 5)
    equation, operands = _common.random_case(rng, min_operands=3, max_operands=3)
    target_path, target_info = target.contract_path(
        equation, *operands, optimize='simple-custom')
    oracle_path, oracle_info = oracle.contract_path(
        equation, *operands, optimize='simple-custom')
    tally.check(int(target_info.opt_cost) == int(oracle_info.opt_cost),
                'custom-path-cost',
                f'{target_info.opt_cost} != {oracle_info.opt_cost}')
    tally.check([tuple(step) for step in target_path]
                == [tuple(step) for step in oracle_path],
                'custom-path-steps', f'{target_path} != {oracle_path}')

    # 3) 未注册的名字必须报错（fail-closed）
    try:
        target.paths.get_path_fn('definitely-not-registered')
    except Exception:  # noqa: BLE001 - 期望失败
        unregistered_rejected = True
    else:
        unregistered_rejected = False
    tally.check(unregistered_rejected, 'unknown-path-rejected',
                '未注册的规划器名未报错')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())

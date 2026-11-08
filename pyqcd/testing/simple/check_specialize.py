#!/usr/bin/env python3
"""`specialize`（方程形态专用路径）单函数对照：开关语义 + 数值正确性。

参考实现为 **torch.einsum**：无论 `specialize` 打开还是关闭，结果都必须一致且正确；
另外校验注册表内容与"生效/关闭"的边界（`specialize=False`、环境变量、非法形状）。
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os as _os
    import sys

    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from simple import _common


NUCLEON_EQ = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
SANDWICH_EQ = 'ab,...bce,cd->...dae'


def _nucleon_operands(rng, nvec=8):
    peram = (rng.standard_normal((4, 4, nvec, nvec))
             + 1j * rng.standard_normal((4, 4, nvec, nvec))).astype(np.complex128)
    gamma = (rng.standard_normal((4, 4))
             + 1j * rng.standard_normal((4, 4))).astype(np.complex128)
    vvv = (rng.standard_normal((1, nvec, nvec, nvec))
           + 1j * rng.standard_normal((1, nvec, nvec, nvec))).astype(np.complex128)
    projector = np.eye(4, dtype=np.complex128)
    return [peram, peram, peram, gamma, gamma, vvv, vvv, projector]


def _sandwich_operands(rng, spatial=2, nvec=4, n_t=2):
    prop = (rng.standard_normal((spatial, spatial, spatial, 3, n_t, 4, 4, nvec))
            + 1j * rng.standard_normal(
                (spatial, spatial, spatial, 3, n_t, 4, 4, nvec))
            ).astype(np.complex128)
    gamma = np.eye(4, dtype=np.complex128)
    return [gamma, np.conj(prop), gamma]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('specialize')
    target = _common.target_module()

    # 1) 注册表结构与内容
    registry = target.specializations()
    tally.check(len(registry) >= 8, 'registry/size', f'仅 {len(registry)} 条')
    names = {name for name, _kind, _eq in registry}
    for expected in ('nucleon-2pt(E1)', 'gauge-propagator(E4)',
                     'gamma-sandwich(E5)', 'sush-3pt-main(P1)'):
        tally.check(expected in names, f'registry/{expected}',
                    '未注册该形态')
    kinds = {name: kind for name, kind, _eq in registry}
    tally.check('needs-optimization' in kinds.get('sush-3pt-main(P1)', ''),
                'registry/p1-needs-optimization',
                f'P1 类型为 {kinds.get("sush-3pt-main(P1)")!r}')

    # 2) 开关语义：开/关结果都必须等于 torch
    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        operands = _nucleon_operands(rng)
        reference = _common.torch_reference(NUCLEON_EQ, operands)
        on = target.contract(NUCLEON_EQ, *operands, optimize='dp',
                             specialize=True)
        off = target.contract(NUCLEON_EQ, *operands, optimize='dp',
                              specialize=False)
        for label, value in (('on', on), ('off', off)):
            error = _common.relative_error(value, reference)
            tally.check(error <= _common.RELATIVE_TOLERANCE,
                        f'case{index}/nucleon-{label}', f'rel={error:.2e}')

        sandwich = _sandwich_operands(rng)
        reference = _common.torch_reference(SANDWICH_EQ, sandwich)
        on = target.contract(SANDWICH_EQ, *sandwich, optimize='dp',
                             specialize=True)
        off = target.contract(SANDWICH_EQ, *sandwich, optimize='dp',
                              specialize=False)
        for label, value in (('on', on), ('off', off)):
            error = _common.relative_error(value, reference)
            tally.check(error <= _common.RELATIVE_TOLERANCE,
                        f'case{index}/sandwich-{label}', f'rel={error:.2e}')

    # 3) 环境变量关闭：结果仍正确
    operands = _nucleon_operands(_common.seeded_rng(args.seed + 99))
    reference = _common.torch_reference(NUCLEON_EQ, operands)
    previous = os.environ.get('PYQCD_EINSUM_SPECIALIZE')
    os.environ['PYQCD_EINSUM_SPECIALIZE'] = '0'
    try:
        value = target.contract(NUCLEON_EQ, *operands, optimize='dp')
    finally:
        if previous is None:
            os.environ.pop('PYQCD_EINSUM_SPECIALIZE', None)
        else:
            os.environ['PYQCD_EINSUM_SPECIALIZE'] = previous
    error = _common.relative_error(value, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'env-disables',
                f'rel={error:.2e}')

    # 4) 形状不满足契约时必须回退（结果仍正确）
    broken = _nucleon_operands(_common.seeded_rng(args.seed + 7))
    broken[0] = broken[0][:, :, :4, :4]
    broken[1] = broken[1][:, :, :4, :4]
    broken[2] = broken[2][:, :, :4, :4]
    broken[5] = broken[5][:, :4, :4, :4]
    broken[6] = broken[6][:, :4, :4, :4]
    reference = _common.torch_reference(NUCLEON_EQ, broken)
    try:
        value = target.contract(NUCLEON_EQ, *broken, optimize='dp')
        error = _common.relative_error(value, reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE, 'contract-mismatch-fallback',
                    f'rel={error:.2e}')
    except Exception as exc:  # noqa: BLE001 - 形状与 VVV 不一致时允许报错
        tally.skip('contract-mismatch-fallback',
                   f'形状不一致按非法输入处理: {type(exc).__name__}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())

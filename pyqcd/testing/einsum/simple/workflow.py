"""真实生产工作流的时间构成对照（2pt + 3pt）。

对照的是 sush 生产脚本的实际循环结构：

* 2pt（`contraction.NN.2pt.py:96-113`）
      for t_src in range(Nt):
          for t_sink in 32 个位移:
              for Mom_indx in ceil(Mom_len/3):
                  contract(E1)
      → 调用数 = Nt × 32 × ceil(Mom_len/3)，Mom_len = 8
* 3pt（`contraction.NJNp-.3pt.cupy.py:125-200`）
      for t_src in range(Nt):
          for t_curr in t_sep+1 个时间片:
              for Mom_indx in ceil(Mom_len/3):
                  2 次 15 算子收缩
                  for Mom_indx2 in ceil(Mom_len/3):
                      3 次 15 算子收缩（P1/P2 交替）
      → 调用数 = Nt × (t_sep+1) × ceil(Mom_len/3) × (2 + 3 × ceil(Mom_len/3))
             = Nt × (t_sep+1) × 3 × 11 = Nt × (t_sep+1) × 33

两种模式：
    --mode model  只测单次中位耗时，再乘以真实调用数得到总时间构成（默认）
    --mode run    真跑全部调用（慢，用于抽检）

用法：
    python -m pyqcd.testing.einsum.simple.workflow
    python -m pyqcd.testing.einsum.simple.workflow --nt 64 --tsep 8 --calls 3
    python -m pyqcd.testing.einsum.simple.workflow --mode run --nt 2 --tsep 1
"""

from __future__ import annotations

import argparse
import math
import os

from . import _common

MOM_LEN = 8
MOM_BLOCK = 3
T_SINK_WINDOW = 32


def counts(nt, tsep, link_max=1):
    """生产调用计数（依据脚本循环结构）。

    * 2pt：`contraction.NN.2pt.py:96-113`  Nt × 32 × ceil(Mom/3)
    * 3pt：`contraction.NJNp-.3pt.cupy.py:125-200`  Nt × (tsep+1) × 3 × (2 + 9)
    * gauge（E4/E5）：`contraction.block.numpy.gen.py:127-195`
      len(t_src)=1 × t_sink 窗 32 × (2*link_max+1)
    """
    per_mom2 = math.ceil(MOM_LEN / MOM_BLOCK)
    two_point = nt * T_SINK_WINDOW * per_mom2
    three_point = nt * (tsep + 1) * MOM_BLOCK * (2 + MOM_BLOCK * 3)
    gauge = 1 * T_SINK_WINDOW * (2 * link_max + 1)
    return two_point, three_point, gauge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='T1')
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--nt', type=int, default=64, help='时间片总数')
    parser.add_argument('--tsep', type=int, default=8, help='3pt 的 t_sep 上限')
    parser.add_argument('--calls', type=int, default=3, help='单次计时样本数')
    parser.add_argument('--mode', choices=('model', 'run'), default='model')
    parser.add_argument('--group2', default='E1', help='2pt 使用的方程')
    parser.add_argument('--group3', default='P1,P2', help='3pt 使用的方程（交替）')
    parser.add_argument('--link-max', type=int, default=1,
                        help='gauge 管线的 link_max（计数 2*link_max+1）')
    parser.add_argument('--gauge-groups', default='E4,E5',
                        help='peram×gauge 管线使用的方程')
    parser.add_argument('--gauge-profile', default=None,
                        help='gauge 管线单独使用的 profile（默认与 --profile 相同）')
    args = parser.parse_args(argv)

    backend = args.backend
    if backend == 'cuda' and not _common.cuda_available():
        print('SKIP: CUDA 不可用')
        return 0
    os.environ.setdefault(
        'PYQCD_EINSUM_NATIVE', 'cuda' if backend == 'cuda' else 'kupl')

    cfg = _common.profile_config(args.profile)
    gauge_names = [name for name in args.gauge_groups.split(',') if name]
    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    backend_name = 'cupy' if backend == 'cuda' else 'numpy'
    two_calls, three_calls, gauge_calls = counts(args.nt, args.tsep, args.link_max)
    groups = [(args.group2, two_calls)]
    three_names = [name for name in args.group3.split(',') if name]
    for name in three_names:
        groups.append((name, three_calls // max(len(three_names), 1)))
    for name in gauge_names:
        groups.append((name, gauge_calls // max(len(gauge_names), 1)))

    print(f'profile={args.profile} backend={backend} mode={args.mode} '
          f'nt={args.nt} tsep={args.tsep}')
    print(f'调用数：2pt={two_calls}  3pt={three_calls}'
          f'（每 (t_src,t_curr) 33 次，Mom_len={MOM_LEN}）'
          f'  gauge={gauge_calls}（link_max={args.link_max}）')
    header = (f'{"group":6s} {"calls":>8s} {"opt_ms/call":>12s} '
              f'{"qcd_ms/call":>12s} {"opt_share":>10s} {"qcd_share":>10s}')
    print(header)
    print('-' * len(header))

    rows = []
    for name, calls in groups:
        meta = _common.EQUATION_BY_NAME.get(name)
        if meta is None:
            print(f'{name}: 未知方程')
            continue
        group_cfg = cfg if name not in gauge_names else _common.profile_config(
            args.gauge_profile or args.profile)
        arrays = _common.operands(name, group_cfg)
        host = _common.to_backend(arrays, backend)
        oracle_path = list(oracle.contract_path(
            meta['eq'], *arrays, optimize='dp')[0])
        target_path = list(target.contract_path(
            meta['eq'], *arrays, optimize='dp')[0])
        if args.mode == 'run':
            samples = max(1, args.calls)
            limit = min(calls, samples)
            opt_total = qcd_total = 0.0
            for _ in range(limit):
                opt_total += _common.median_seconds(
                    lambda: oracle.contract(meta['eq'], *host,
                                            optimize=oracle_path,
                                            backend=backend_name), 1, 0, backend)
                qcd_total += _common.median_seconds(
                    lambda: target.contract(meta['eq'], *host,
                                            optimize=target_path,
                                            backend=backend_name), 1, 0, backend)
            opt_per_call = opt_total / limit * 1e3
            qcd_per_call = qcd_total / limit * 1e3
        else:
            opt_per_call = _common.median_seconds(
                lambda: oracle.contract(meta['eq'], *host,
                                        optimize=oracle_path,
                                        backend=backend_name),
                max(args.calls, 3), 1, backend) * 1e3
            qcd_per_call = _common.median_seconds(
                lambda: target.contract(meta['eq'], *host,
                                        optimize=target_path,
                                        backend=backend_name),
                max(args.calls, 3), 1, backend) * 1e3
        rows.append((name, calls, opt_per_call, qcd_per_call))
        del arrays, host

    opt_total = sum(row[1] * row[2] for row in rows) or 1.0
    qcd_total = sum(row[1] * row[3] for row in rows) or 1.0
    for name, calls, opt_ms, qcd_ms in rows:
        print(f'{name:6s} {calls:8d} {opt_ms:12.2f} {qcd_ms:12.2f} '
              f'{calls * opt_ms / opt_total * 100:9.1f}% '
              f'{calls * qcd_ms / qcd_total * 100:9.1f}%')

    print('-' * len(header))
    print(f'{"total":6s} {"":8s} {opt_total / 1e3:12.1f}s '
          f'{qcd_total / 1e3:12.1f}s   倍率={opt_total / qcd_total:.2f}x')
    print('\n注：model 模式的时间为「单次中位 × 调用数」的推算，'
          '不是整段工作流实测；run 模式才整段实跑。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

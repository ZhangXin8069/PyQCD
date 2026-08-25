"""对照单测编排入口：--group all|lqcddb|donghx [--only id,id]。"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from harness import run_case, save_results


def build_cases(group):
    cases = []
    if group in ('all', 'lqcddb'):
        import cases_lqcddb
        import cases_lqcddb2
        cases += cases_lqcddb.build()
        cases += cases_lqcddb2.build()
        import cases_suppl
        cases += cases_suppl.build()
    if group in ('all', 'donghx'):
        import cases_donghx
        import cases_donghx2
        cases += cases_donghx.build()
        cases += cases_donghx2.build()
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='all',
                    choices=['all', 'lqcddb', 'donghx'])
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    cases = build_cases(args.group)
    if args.only:
        ids = set(args.only.split(','))
        cases = [c for c in cases if c.cid in ids]
    results = []
    for c in cases:
        print(f"[{c.group}] {c.cid}: {c.desc}", flush=True)
        rec = run_case(c)
        print(f"   -> {rec['status']} diff={rec.get('diff', '-')} "
              f"t_ref={rec.get('t_ref')} t_pq={rec.get('t_pq')}", flush=True)
        results.append(rec)
    outdir = os.path.join(_HERE, 'v' + time.strftime('%Y%m%d%H%M'))
    save_results(results, outdir)
    n_pass = sum(1 for r in results if r['status'] == 'pass')
    print(f"== PASS {n_pass}/{len(results)} -> {outdir}")


if __name__ == '__main__':
    main()

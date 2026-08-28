"""cmp1 断言门：校验最近一次 results.json 的通过率与关键项。

用法：python examples/pyqcd/cmp1/verify_cmp1.py [results.json]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 允许非 pass 的白名单（结构性/统计性/登记差异项）
WHITELIST_STRUCTURAL = {'L17', 'L20', 'L22b', 'L28', 'L30', 'D08',
                        'S10'}
# 数值分歧已登记 MAPPING.md optim/backlog 的项（不要求 note 字段）
WHITELIST_REGISTERED = {'S09', 'D04'}
# 数值分歧已登记 optim/backlog 的项：要求 status=diff 且已附 note
WHITELIST_DIFF_NOTE = {'L25', 'S09'}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob(os.path.join(HERE, 'v*', 'results.json')))[-1]
    results = json.load(open(path))
    fails = []
    for r in results:
        st, cid = r['status'], r['id']
        if st == 'pass':
            continue
        if st == 'diff' and cid in WHITELIST_STRUCTURAL:
            continue
        if st == 'diff' and cid in WHITELIST_DIFF_NOTE and r.get('note'):
            continue
        if st == 'diff' and cid in WHITELIST_REGISTERED:
            continue
        fails.append(f"{cid} {st} {r.get('diff', '')}")
    n_pass = sum(1 for r in results if r['status'] == 'pass')
    print(f"results: {path}")
    print(f"pass={n_pass}/{len(results)} whitelist_ok="
          f"{len(results) - n_pass - len(fails)} hard_fails={len(fails)}")
    for f_ in fails:
        print('  HARD-FAIL:', f_)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()

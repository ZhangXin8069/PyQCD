"""Conservative, recoverable cleanup of this one HPC workspace.

Only source/config metadata are inspected. No physics imports, tensor loading,
Slurm mutations, generic cache-name deletion, or changes outside ROOT occur.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / '_archive' / '20261027'
PLAN = HERE / 'cleanup_plan.json'
RECEIPT = HERE / 'cleanup_receipt.json'
FORBIDDEN = {'.git', '.codex', '.agents', '.aws', '_archive', HERE.name}
ACTIVE = {
    'quark_l32x64_B_expand50_P456_20261026',
    'quark_l32x96_B_matched_radius20_7flow_20261021',
    'quark_l32x96_B_matched_radius20_7flow_retry_v2_20261021',
}
# Explicit exceptions to the textual graph: production data and reusable
# projectors may be used from outside this workspace. Leave them in place.
EXTRA_KEEP = {
    'build_L32x64_c3c2.py': 'gluon shared scientific code; outside quark cleanup',
    'collect_L32x64_ratio_fits.py': 'gluon shared scientific code',
    'collect_L32x64_ratio_fits.slurm': 'gluon shared scientific code',
    'plot_L32x64_barefit.py': 'gluon shared plotting code',
    'plot_L32x64_barefit.slurm': 'gluon shared plotting code',
    'plot_current_gluon_flow_results.py': 'gluon shared plotting code',
    'submit_L32x64_P6_helicity.slurm': 'gluon historical production entry',
    'submit_L32x64_P6_unpolarized.slurm': 'gluon historical production entry',
    'submit_L32x64_helicity.slurm': 'gluon historical production entry',
    'submit_L32x64_unpolarized.slurm': 'gluon historical production entry',
    'quark_project_channels_local.py': 'shared spin/channel projector; external users possible',
    'quark_l32x64_expand40_20261013': 'historical production campaign; preserve code/data provenance',
    'quark_spin2_missing30_no_tf1_20260928_v2': 'historical production recovery receipts',
    'quark_spin2_missing30_no_tf1_20260928_v3': 'historical production recovery receipts',
    'quark_spin2_missing30_no_tf1_20260929_v4': 'historical production recovery receipts',
    'quark_spin2_production_20260922': 'original production and regression data',
    'quark_spin2_tf03_tf04_20261004': 'historical production entry and cleanup receipts',
}
ANCHOR_TREES = [
    'quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2/fits',
    'quark_l32x64_B_Pz0_ratio20_boot100_T7_Q1em3_chi1e3_20261021/fits',
    'quark_l32x64_B_pionDA_workflow_20261022/00_audit',
    'quark_l32x64_B_pionDA_workflow_20261022/01_bare',
    'quark_l32x64_B_pionDA_workflow_20261022/02_localnorm',
    'quark_l32x64_B_pionDA_workflow_20261022/03_line',
    'quark_l32x64_B_m0_RGR_z12_20261025/04_m0',
    'quark_l32x64_B_moving_RGR_m0_20261025/04_flow_conversion',
    'quark_l32x64_B_moving_RGR_m0_20261025/05_tau0_linear',
    'quark_l32x64_B_moving_RGR_m0_20261025/06_m0_corrected',
    'quark_l32x64_B_moving_RGR_m0_20261025/data',
    'quark_l32x64_B_moving_RGR_m0_20261025/plots',
    'quark_l32x64_B_06_lambda_fourier_lamet_v2_20261025/results',
    'quark_l32x64_B_06_lambda_fourier_lamet_v2_20261025/lhapdf_comparison_20261025',
]


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_save(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def checked_path(relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    if rel.parts[0] in {'.git', '.codex', '.agents', '.aws'}:
        raise ValueError(f'Protected special path: {relative}')
    path = ROOT / rel
    # Never traverse a symlink parent into another tree.
    for parent in path.parents:
        if parent == ROOT:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink parent: {relative}')
    return path


def metadata(path):
    st = path.lstat()
    return dict(path=str(path.relative_to(ROOT)), inode=st.st_ino,
                device=st.st_dev, size=st.st_size, mtime_ns=st.st_mtime_ns,
                allocated_bytes=getattr(st, 'st_blocks', 0) * 512,
                kind=('symlink' if stat.S_ISLNK(st.st_mode) else
                      'directory' if stat.S_ISDIR(st.st_mode) else 'file'))


def walk_files(path):
    if path.is_symlink() or path.is_file():
        yield path
        return
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            if (base / name).is_symlink():
                yield base / name
        dirs[:] = [d for d in dirs if not (base/d).is_symlink()]
        for name in files:
            yield base / name


def corrected_graph(inventory):
    """Reuse the completed read-only scan, conservatively repairing stem aliases.

    An old directory and a Python script can share a stem. Retain both for an
    ambiguous token; do not interpret the directory as an unused product.
    """
    names = {row['entry'] for row in inventory['inventory']}
    graph = {name: set() for name in names}
    edges = []
    for edge in inventory['edges']:
        graph[edge['source']].add(edge['target'])
        edges.append(edge)
        target = edge['target']
        if target.endswith('.py') and target[:-3] in names:
            alternate = dict(edge, target=target[:-3], ambiguity='directory/python shared stem')
            graph[edge['source']].add(alternate['target'])
            edges.append(alternate)
    # The initial scan discarded self references, which can hide a script's
    # same-stem output directory. Re-read only the loose Python scripts.
    for name in names:
        if name.endswith('.py') and name[:-3] in names:
            text = (ROOT/name).read_text(errors='replace')
            if re.search(r'(?<![\w.-])' + re.escape(name[:-3]) + r'(?![\w.-])', text):
                graph[name].add(name[:-3])
                edges.append(dict(source=name, target=name[:-3], evidence=name,
                                  ambiguity='directory/python shared stem'))
    protected = set(inventory['preserved_entries']) | set(EXTRA_KEEP) | ACTIVE
    while True:
        new = set().union(*(graph[name] for name in protected)) - protected
        if not new:
            break
        protected.update(new)
    return protected, edges


def archive_group(name):
    if name.startswith('quark_ringed_recovery_'):
        return 'superseded_solver_trials'
    if name.endswith(('.py', '.slurm', '.sh')):
        return 'legacy_quark_scripts'
    if name.startswith(('test_', 'slurm-')) or name == 'package-lock.json':
        return 'old_logs_and_misc'
    return 'legacy_quark_analysis'


def cache_plan(inventory):
    parents = {ROOT}
    parents.update(ROOT/x['entry'] for x in inventory['inventory'] if x['kind'] == 'directory')
    parents.update((ROOT/x['path']).parent for x in inventory['code_and_manifest_fingerprints'])
    cache_files = {}
    for parent in sorted(parents):
        relative = parent.relative_to(ROOT)
        if relative.parts and relative.parts[0] in (ACTIVE | FORBIDDEN):
            continue
        for cache_name in ('__pycache__', '.pytest_cache', 'matplotlib_cache'):
            cache = parent/cache_name
            if not cache.is_dir() or cache.is_symlink():
                continue
            candidates = list(walk_files(cache))
            if any(p.is_symlink() for p in candidates):
                continue
            if cache_name == '__pycache__' and any(p.suffix != '.pyc' for p in candidates):
                continue
            if cache_name == 'matplotlib_cache':
                candidates = [p for p in candidates if re.fullmatch(r'fontlist-v\d+\.json', p.name)]
            for path in candidates:
                if path.is_file():
                    record = metadata(path)
                    record.update(reason=cache_name + ' regenerable cache', sha256=sha(path))
                    cache_files[record['path']] = record
    for directory in ('docs_update_20261009_YyHRbs', 'ringed_doc_update_20260920_oygtSM/note'):
        base = ROOT/directory
        for suffix in ('.aux', '.blg', '.log', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz'):
            path = base/('main' + suffix)
            if path.is_file() and not path.is_symlink():
                record = metadata(path)
                record.update(reason='old temporary TeX build intermediate; source/PDF/bbl retained', sha256=sha(path))
                cache_files[record['path']] = record
    return sorted(cache_files.values(), key=lambda x: x['path'])


def plan():
    if RECEIPT.exists():
        raise RuntimeError('A receipt already exists; do not re-plan an executed cleanup.')
    inventory = json.loads((HERE/'inventory.json').read_text())
    protected, edges = corrected_graph(inventory)
    candidates = sorted(set(inventory['unreferenced_entries']) - protected - {'__pycache__', '.pytest_cache'})
    moves = []
    deleted_cache = cache_plan(inventory)
    removed_paths = {row['path'] for row in deleted_cache}
    archive_files = []
    for name in candidates:
        if name in FORBIDDEN or '/' in name:
            raise ValueError(f'Unexpected archive target: {name}')
        path = checked_path(name)
        record = metadata(path)
        record.update(destination=str((ARCHIVE/archive_group(name)/name).relative_to(ROOT)),
                      reason='unreferenced by retained source/config/handoff graph; historical artifact, archive only')
        moves.append(record)
        print('PLAN_ARCHIVE', name, flush=True)
        for item in walk_files(path):
            if item.name == 'AGENTS.md':
                raise RuntimeError(f'Read additional instructions before proceeding: {item}')
            item_record = metadata(item)
            if item_record['path'] in removed_paths:
                continue
            if item_record['kind'] == 'file' and item.suffix in {'.py', '.sh', '.slurm', '.json', '.md', '.tex'}:
                item_record['sha256'] = sha(item)
            archive_files.append(item_record)
    anchors = []
    for relative in ANCHOR_TREES:
        base = checked_path(relative)
        if not base.is_dir():
            raise FileNotFoundError(f'Mainline product directory missing: {relative}')
        for item in walk_files(base):
            if item.suffix.lower() in {'.npz', '.npy', '.pdf', '.png', '.h5', '.hdf5', '.csv'} and not item.is_symlink():
                anchors.append(metadata(item))
    top_records = [metadata(ROOT/name) for name in sorted(protected)
                   if name not in (FORBIDDEN | {'__pycache__', '.pytest_cache'})]
    result = dict(generated=now(), root=str(ROOT), handoff=inventory['handoff'],
                  handoff_sha256=inventory['handoff_sha256'],
                  preserved_entries=sorted(protected), conservative_exceptions=EXTRA_KEEP,
                  archive_moves=moves, archive_files=archive_files,
                  cache_deletions=deleted_cache, product_anchors=anchors,
                  protected_top_entries=top_records,
                  code_and_manifest_fingerprints=inventory['code_and_manifest_fingerprints'],
                  active_job_roots=sorted(ACTIVE),
                  description='Recoverable same-filesystem rename; only regenerable non-science caches are deleted.')
    save(PLAN, result)
    csv_save(HERE/'archive_manifest.csv', moves, ('path', 'destination', 'kind', 'reason', 'inode', 'device'))
    csv_save(HERE/'cache_deletions.csv', deleted_cache, ('path', 'reason', 'size', 'allocated_bytes', 'sha256'))
    csv_save(HERE/'protected_products.csv', anchors, ('path', 'size', 'mtime_ns', 'inode', 'device'))
    csv_save(HERE/'retained_entries.csv', [dict(entry=name, reason=EXTRA_KEEP.get(name, 'handoff/source/config dependency closure'))
                                         for name in sorted(protected)], ('entry', 'reason'))
    save(HERE/'corrected_dependency_edges.json', edges)
    print(json.dumps(dict(archive_entries=len(moves), archived_files=len(archive_files),
        archived_logical_bytes=sum(x['size'] for x in archive_files if x['kind'] == 'file'),
        delete_cache_files=len(deleted_cache),
        delete_cache_logical_bytes=sum(x['size'] for x in deleted_cache),
        delete_cache_allocated_bytes=sum(x['allocated_bytes'] for x in deleted_cache),
        protected_products=len(anchors), retained_top_entries=len(protected)), indent=2), flush=True)


def expect_same(record, path, *, directory=False):
    actual = metadata(path)
    fields = ('inode', 'device') if directory else ('inode', 'device', 'size', 'mtime_ns')
    if any(actual[k] != record[k] for k in fields):
        raise RuntimeError(f'Object changed since plan: {record["path"]}')
    if record.get('sha256') and sha(path) != record['sha256']:
        raise RuntimeError(f'Hash changed: {record["path"]}')


def execute():
    if RECEIPT.exists():
        raise RuntimeError('This cleanup already has a receipt; refusing repeat execution.')
    data = json.loads(PLAN.read_text())
    if str(ROOT) != data['root'] or sha(data['handoff']) != data['handoff_sha256']:
        raise RuntimeError('Workspace/handoff changed; re-audit required.')
    for record in data['archive_moves']:
        source = checked_path(record['path'])
        destination = checked_path(record['destination'])
        if record['path'] in data['preserved_entries'] or destination.exists() or destination.is_symlink():
            raise RuntimeError(f'Protected/occupied move: {source}')
        expect_same(record, source, directory=record['kind'] == 'directory')
    # Preflight all cache deletions before touching any object.
    for record in data['cache_deletions']:
        expect_same(record, checked_path(record['path']))
    for record in data['code_and_manifest_fingerprints']:
        if sha(checked_path(record['path'])) != record['sha256']:
            raise RuntimeError(f'Frozen source/manifest changed: {record["path"]}')
    receipt = dict(started=now(), root=str(ROOT), plan_sha256=sha(PLAN),
                   archive_moves=[], cache_deleted=[], empty_cache_directories_removed=[])
    save(RECEIPT, receipt)
    for record in data['cache_deletions']:
        checked_path(record['path']).unlink()
        receipt['cache_deleted'].append(record)
    cache_parents = set()
    for record in data['cache_deletions']:
        parent = checked_path(record['path']).parent
        while parent != ROOT:
            if parent.name in {'__pycache__', '.pytest_cache', 'matplotlib_cache'}:
                cache_parents.add(parent)
                break
            if '.pytest_cache' in parent.parts:
                cache_parents.add(parent)
            parent = parent.parent
    for directory in sorted(cache_parents, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()  # Only remove an empty, specifically inventoried cache.
            receipt['empty_cache_directories_removed'].append(str(directory.relative_to(ROOT)))
        except OSError:
            pass
    save(RECEIPT, receipt)
    for record in data['archive_moves']:
        source, destination = checked_path(record['path']), checked_path(record['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Path.rename never copies or rewrites data; the original inode remains.
        source.rename(destination)
        receipt['archive_moves'].append(record)
        save(RECEIPT, receipt)
        print('ARCHIVED', record['path'], flush=True)
    receipt['finished'] = now()
    receipt['archived_regular_file_bytes'] = sum(x['size'] for x in data['archive_files'] if x['kind'] == 'file')
    receipt['cache_logical_bytes_deleted'] = sum(x['size'] for x in receipt['cache_deleted'])
    receipt['cache_allocated_bytes_deleted'] = sum(x['allocated_bytes'] for x in receipt['cache_deleted'])
    save(RECEIPT, receipt)
    verify()


def moved_path(path, moves):
    for row in moves:
        if path == row['path'] or path.startswith(row['path'] + '/'):
            return checked_path(row['destination'] + path[len(row['path']):])
    return checked_path(path)


def verify():
    data = json.loads(PLAN.read_text())
    receipt = json.loads(RECEIPT.read_text())
    failures = []

    def check(label, function):
        try:
            function()
        except Exception as exc:
            failures.append(dict(check=label, error=str(exc)))

    check('handoff hash', lambda: (_ for _ in ()).throw(RuntimeError('handoff changed'))
          if sha(data['handoff']) != data['handoff_sha256'] else None)
    for row in data['protected_top_entries']:
        check(row['path'], lambda row=row: expect_same(row, checked_path(row['path']), directory=row['kind'] == 'directory'))
    for row in data['code_and_manifest_fingerprints']:
        path = moved_path(row['path'], receipt['archive_moves'])
        check(row['path'], lambda row=row, path=path: (_ for _ in ()).throw(RuntimeError('source hash changed'))
              if sha(path) != row['sha256'] else None)
    for row in data['product_anchors']:
        check(row['path'], lambda row=row: expect_same(row, checked_path(row['path'])))
    for row in data['archive_moves']:
        check(row['destination'], lambda row=row: expect_same(row, checked_path(row['destination']), directory=row['kind'] == 'directory'))
        if checked_path(row['path']).exists():
            failures.append(dict(check=row['path'], error='old archive target still exists'))
    for row in data['archive_files']:
        path = moved_path(row['path'], receipt['archive_moves'])
        check(row['path'], lambda row=row, path=path: expect_same(row, path))
    for row in data['cache_deletions']:
        if moved_path(row['path'], receipt['archive_moves']).exists():
            failures.append(dict(check=row['path'], error='planned cache still exists'))
    result = dict(generated=now(), status='PASS' if not failures else 'FAIL',
        archive_entries=len(receipt['archive_moves']), archived_file_metadata_checks=len(data['archive_files']),
        preserved_source_hash_checks=len(data['code_and_manifest_fingerprints']),
        protected_product_metadata_checks=len(data['product_anchors']),
        protected_top_entries=len(data['protected_top_entries']),
        cache_files_deleted=len(receipt['cache_deleted']), failures=failures,
        limits='No numerical physics tests. Large scientific arrays were verified by unchanged inode/size/mtime, not content reloading.')
    save(HERE/'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError('Cleanup verification failed; inspect verification.json.')


def restore(names, do_restore):
    receipt = json.loads(RECEIPT.read_text())
    selected = receipt['archive_moves'] if not names else [r for r in receipt['archive_moves'] if r['path'] in names]
    if names and len(selected) != len(set(names)):
        raise ValueError('Unknown archive entry; use an original top-level name from archive_manifest.csv.')
    # Refuse to overwrite any current object, even a broken symlink.
    for row in selected:
        target, source = checked_path(row['path']), checked_path(row['destination'])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Restore would overwrite {target}')
        expect_same(row, source, directory=row['kind'] == 'directory')
    for row in selected:
        print('RESTORE' if do_restore else 'WOULD_RESTORE', row['destination'], '->', row['path'], flush=True)
        if do_restore:
            checked_path(row['destination']).rename(checked_path(row['path']))
    if do_restore:
        save(HERE/('restoration_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'),
             dict(generated=now(), restored=[r['path'] for r in selected], caches='regenerate from source; not archived'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'execute', 'verify', 'restore'))
    parser.add_argument('--names', nargs='*', default=[])
    parser.add_argument('--execute-restore', action='store_true', help='restore is dry-run unless explicitly supplied')
    args = parser.parse_args()
    if args.action == 'plan':
        plan()
    elif args.action == 'execute':
        execute()
    elif args.action == 'verify':
        verify()
    else:
        restore(args.names, args.execute_restore)


if __name__ == '__main__':
    main()

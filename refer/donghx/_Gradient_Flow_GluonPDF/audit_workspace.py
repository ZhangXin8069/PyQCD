"""Read-only dependency inventory before a recoverable workspace cleanup.

Never import physics code or open lattice arrays. Examine small source/config
files, record edges between top-level entries, and preserve immutable jobs.
"""
from __future__ import annotations

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
HANDOFF = Path('/public/home/donghx/.codex/Markdown/gradient_flow_quark_pdf.md')
EXCLUDED = {'.git', '.codex', '.agents', '.aws', '__pycache__', '.pytest_cache',
            'basis_cache', 'flow_cache', 'gauge_cache', 'basis', 'flowed_gauge',
            'vertex', 'vertex_schemeB', 'vvv_cache', 'VVV', 'perambulators',
            'eigensystem', 'C3_C2_schemeB', 'C3_schemeB_Pz3456_tsep7to12',
            'output', 'output_v5', 'output_L32x64_v1', 'logs', 'results',
            'data', 'plots', 'figures', 'figs', 'operators', 'reference_data',
            'smoke', 'smoke_two_endpoint_preflight', 'status', 'summary',
            'scratch', 'input_stage', '_archive', 'workspace_cleanup_20261027'}
SUFFIXES = {'.py', '.sh', '.slurm', '.json', '.md', '.txt', '.cpp', '.c', '.h', '.tex'}
MAX_FILE_BYTES = 40 * 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def source_files(top: Path):
    if top.is_symlink():
        return
    if top.is_file():
        if top.suffix in SUFFIXES and top.stat().st_size <= MAX_FILE_BYTES:
            yield top
        return
    if top.name in EXCLUDED:
        return
    for directory, dirs, files in os.walk(top, followlinks=False):
        base = Path(directory)
        depth = len(base.relative_to(top).parts)
        dirs[:] = [name for name in dirs if name not in EXCLUDED
                   and not (base/name).is_symlink() and depth < 4]
        for name in files:
            path = base/name
            if path.suffix in SUFFIXES and not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES:
                yield path


def main():
    entries = sorted(p for p in ROOT.iterdir() if p.name not in ('.git', '.codex', '.agents', '.aws'))
    names = {p.name for p in entries}
    tokens = {name: {name} for name in names}
    # Capture Python imports of loose scripts as well as literal filenames.
    for path in entries:
        if path.suffix == '.py':
            tokens.setdefault(path.stem, set()).add(path.name)
    expression = re.compile(r'(?<![\w.-])(' + '|'.join(map(re.escape, sorted(tokens, key=len, reverse=True))) + r')(?![\w.-])')

    def references(text):
        return set().union(*(tokens[m.group(1)] for m in expression.finditer(text)))

    note = HANDOFF.read_text()
    direct = references(note)
    # The handoff predates the completed 07--09 follow-up and latest production.
    additions = {
        'quark_l32x64_B_06_lambda_fourier_lamet_v2_20261025': 'newer completed lambda/Fourier/LaMET successor',
        'quark_l32x64_B_expand50_P456_20261026': 'active 50-configuration GPU production and CPU audit',
        'quark_l32x96_B_matched_radius20_7flow_retry_v2_20261021': 'active L32x96 GPU production',
        'quark_l32x96_B_matched_radius20_7flow_20261021': 'queued predecessor audit and provenance',
    }
    preserve_other_science = {
        'README.md', '.gitignore', 'src', 'scripts', 'slurm', 'build', 'tests',
        'analysis', 'comparison', 'matching', 'output', 'output_v5', 'output_L32x64_v1',
        'manifest.tsv', 'manifest_L32x64.tsv', 'logs', 'handoff', 'plans',
        'documentation_checks', 'documentation_updates', 'jobs', 'runs',
        'workspace_cleanup_20261027',
    }
    preserve_other_science.update(n for n in names if n.startswith((
        'gluon_', 'line_mass_', 'wilson_', 'static_potential_', 'polyakov_',
        'fit_', 'flow_time_', 'flow_extrap_', 'ratio_bootstrap_', '.gluon_', '.ratio_')))
    initial = (direct | set(additions) | preserve_other_science) & names
    reasons = {n: 'named handoff reference' for n in direct}
    reasons.update(additions)
    for n in preserve_other_science:
        reasons.setdefault(n, 'preserve gluon/loop science, shared tooling, or current documentation')
    graph = {n: set() for n in names}
    edges, scanned, instructions, fingerprints = [], [], [], []
    for top in entries:
        if top.name == HERE.name:
            continue
        count = 0
        for path in source_files(top):
            content = path.read_text(errors='replace')
            targets = references(content) - {top.name}
            graph[top.name].update(targets)
            for target in sorted(targets):
                edges.append(dict(source=top.name, target=target, evidence=str(path.relative_to(ROOT))))
            if path.name == 'AGENTS.md':
                instructions.append(str(path.relative_to(ROOT)))
            if (top.name in initial and path.suffix in {'.py', '.slurm', '.sh'}) or path.name in {
                    'source_manifest.json', 'manifest.json', 'request.json', 'config.json', 'complete.json'}:
                fingerprints.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size))
            count += 1
        scanned.append(dict(entry=top.name, scanned_sources=count))
        print('SOURCE_SCAN', top.name, count, flush=True)
    protected = set(initial)
    while True:
        new = set().union(*(graph[n] for n in protected)) - protected
        if not new:
            break
        for n in new:
            parents = sorted(p for p in protected if n in graph[p])
            reasons[n] = 'transitive dependency of ' + ', '.join(parents)
        protected.update(new)
    inventory = []
    for path in entries:
        st = path.lstat()
        kind = 'symlink' if stat.S_ISLNK(st.st_mode) else 'directory' if path.is_dir() else 'file'
        inventory.append(dict(entry=path.name, kind=kind, protected=path.name in protected,
            reason=reasons.get(path.name, 'unreferenced historical item; inspect before archive'),
            top_inode=st.st_ino, top_mtime_ns=st.st_mtime_ns,
            file_bytes=st.st_size if kind == 'file' else None))
    result = dict(generated=datetime.now().astimezone().isoformat(), root=str(ROOT),
        handoff=str(HANDOFF), handoff_sha256=sha(HANDOFF), direct_handoff=sorted(direct),
        preserved_entries=sorted(protected), unreferenced_entries=sorted(names-protected),
        inventory=inventory, scanned=scanned, instruction_files=instructions,
        edges=edges, code_and_manifest_fingerprints=fingerprints,
        method='Top-level textual references and Python import names; uncertain items require review. No tensor reads.')
    dump(HERE/'inventory.json', result)
    with (HERE/'inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    with (HERE/'dependency_edges.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('source', 'target', 'evidence'))
        writer.writeheader(); writer.writerows(edges)
    print('PRESERVED', len(protected), 'UNREFERENCED', len(names-protected), flush=True)
    print('\n'.join(sorted(names-protected)), flush=True)


if __name__ == '__main__':
    main()

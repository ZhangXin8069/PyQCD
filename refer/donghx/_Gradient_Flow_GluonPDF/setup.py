"""Freeze isolated successors for the matched-radius front-50 supplement."""
import csv
import hashlib
import json
import re
from pathlib import Path
from shutil import copy2, copytree
from math import ceil, sqrt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAMP = '20261127'
REF_FLOWS = (.9, .97)
REF_WINDOWS = (9, 9)
REFERENCE_A = .0897
CAMPAIGNS = {
    'C24': ('l24x72', .105, (3, -3, 4, -4, 5, -5), 'quark_l24x72_B_50conf_matched_radius_20261104_v2'),
    'E32': ('l32x64', .0897, (3, -3, 4, -4, 5, -5, 6, -6), 'quark_l32x64_B_conf101to150_P0456_20261125/moving'),
    'F32': ('l32x96', .0775, (3, -3, 4, -4, 5, -5), 'quark_l32x96_B_expand50_matched_radius_20261028'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


def setting(text, name, value):
    text, count = re.subn(rf'^{name}\s*=.*$', f'{name} = {value!r}', text, flags=re.M)
    assert count == 1, name
    return text


def config_path(key, source):
    if key == 'C24':
        return source / 'confs.txt'
    if key == 'E32':
        return ROOT / 'quark_l32x64_B_expand50_P456_20261026/combined_confs50.txt'
    return ROOT / 'quark_l32x96_B_expand70_matched_radius_20261109/combined_confs70.txt'


def create(key):
    lattice, spacing, momenta, template = CAMPAIGNS[key]
    source = ROOT / template
    original = config_path(key, source)
    confs = tuple(original.read_text().split()[:50])
    assert len(set(confs)) == 50 and all(conf.isdigit() for conf in confs)
    target = ROOT / f'quark_{lattice}_B_rf025_front50_{STAMP}'
    target.mkdir()
    for subdir in ('logs', 'status', 'locks', 'summary'):
        (target / subdir).mkdir()
    (target / 'confs.txt').write_text('\n'.join(confs) + '\n')
    copytree(source / 'reference', target / 'reference', ignore=lambda _, names: {'__pycache__'} & set(names))
    files = ('common.py', 'generate_basis.py', 'gpu_worker.py', 'launch.py', 'audit_cpu.py', 'flow_safety.py')
    if key == 'E32':
        files += ('insertion_kernels.py',)
    else:
        files += ('test_cpu.py', 'test_retry_cpu.py', 'epsilon_gate.py')
    if key == 'C24':
        files += ('gpu_validation.py', 'backend_sample.py')
    for name in files:
        copy2(source / name, target / name)
    for name in ('run_gpu.slurm', 'run_cpu.slurm', 'prepare_inputs.py'):
        copy2(HERE / name, target / name)
    patch = target / 'common.py'
    text = setting(setting(patch.read_text(), 'MOMENTA', momenta), 'CONFS', confs)
    if key != 'E32':
        text = setting(setting(text, 'FLOWS64', REF_FLOWS), 'WINDOWS64', REF_WINDOWS)
        text = setting(text, 'PROFILE', 'Pm345_zline_open_current_spin_three_channels_rf025')
        text = setting(text, 'SCHEMA', f'{lattice}_B_rf025_Pm345_v1')
    else:
        text = setting(setting(text, 'FLOWS', REF_FLOWS), 'WINDOWS', REF_WINDOWS)
        text = setting(text, 'SCHEMA', 'L32x64_B_rf025_Pm3456_v1')
    patch.write_text(text)
    if key != 'E32':
        for name in ('launch.py', 'epsilon_gate.py', 'backend_sample.py'):
            if not (target / name).exists():
                continue
            path = target / name
            text = path.read_text().replace('(0,6)', '(0,1)').replace('[0,6]', '[0,1]')
            text = text.replace("load_manifest()['tasks'][6]", "load_manifest()['tasks'][1]")
            path.write_text(text)
        path = target / 'test_cpu.py'
        text, count = re.subn(r'assert MOMENTA == \([^\n]*\)', f'assert MOMENTA == {momenta!r}', path.read_text())
        assert count == 1
        path.write_text(text)
    else:
        path = target / 'audit_cpu.py'
        text = path.read_text().replace('== 400', '== 100').replace('100800', '25200')
        text = text.replace('COMPLETE_50_NEW_CONF_8_FLOW_B_P456', 'COMPLETE_FRONT50_TWO_FLOW_B_P3456')
        text = text.replace('configuration_count=50, previous_configuration_count=50, combined_configuration_count=100,',
                            'configuration_count=50,').replace('flow_count=8', 'flow_count=2')
        path.write_text(text)
    if key == 'C24':
        copy2(source / 'status/input_probe.json', target / 'status/input_probe.json')
    schedule = []
    for reference, ref_window in zip(REF_FLOWS, REF_WINDOWS):
        t_f = reference * REFERENCE_A**2
        epsilon = .01 if key == 'E32' else .005 * (REFERENCE_A / spacing)**2
        tau = (round(reference / .005) * epsilon) if key != 'E32' else reference
        assert abs(tau * spacing**2 - t_f) < 1e-16 and sqrt(8*t_f) < .25
        schedule.append(dict(tau64=reference, flow_time_a2=tau, tf_fm2=t_f, radius_fm=sqrt(8*t_f),
                             epsilon=epsilon, nstep=round(tau/epsilon), window64=ref_window,
                             window=ceil(ref_window * REFERENCE_A/spacing)))
    with (target / 'flow_schedule.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(schedule[0]))
        writer.writeheader()
        writer.writerows(schedule)
    source_hashes = {str(path.relative_to(target)): digest(path) for path in target.rglob('*')
                     if path.is_file() and ('status' not in path.relative_to(target).parts)}
    record = dict(ensemble=key, a_fm=spacing, source=str(source), confs=list(confs),
                  momenta=list(momenta), schedule=schedule, files_sha256=source_hashes,
                  upstream_sha256={str(original): digest(original)})
    if key == 'C24':
        record['input_probe_sha256'] = digest(target / 'status/input_probe.json')
        record['backend_library_sha256'] = digest(ROOT / 'quark_ringed_kinetic_v1/quda_zero_guard/build/libquda.so')
    json_write(target / 'request.json', record)
    print(key, target, confs[0], confs[-1], momenta, schedule)


if __name__ == '__main__':
    for ensemble in CAMPAIGNS:
        create(ensemble)

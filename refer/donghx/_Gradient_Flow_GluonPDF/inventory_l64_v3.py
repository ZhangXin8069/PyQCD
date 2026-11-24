"""Compare only recorded fingerprint fields, matching original input guards."""
import inventory_cpu as base

_checked = {}


def inputs_guard(module, manifest):
    containers = (manifest['configuration_inputs'].values() if 'configuration_inputs' in manifest
                  else (row['inputs'] for row in manifest['tasks']))
    expected = {}
    for container in containers:
        for record in container:
            previous = expected.setdefault(record['path'], record)
            assert previous == record, ('inconsistent input record', record['path'])
    for index, (path, record) in enumerate(expected.items()):
        actual = _checked.get(path)
        if actual is None or not set(record).issubset(actual):
            actual = module.fingerprint(path)
            _checked[path] = actual
        # Some old manifests omit data_offset; p0.fingerprint returns it.
        # Preserve EVERY declared field rather than require absent fields.
        for key, value in record.items():
            assert actual.get(key) == value, ('input changed', path, key, actual.get(key), value)
        if index % 500 == 0:
            print('INPUT_DECLARED_FIELDS_PASS', module.HERE.name, index, len(expected), flush=True)
    return len(expected)


if __name__ == '__main__':
    for path, digest in base.read(base.ROOT / 'l64_v3_lock.json').items():
        assert base.sha(path) == digest, path
    base.inputs_guard = inputs_guard
    base.main()

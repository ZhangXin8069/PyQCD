"""L64 inventory adapter: its moving manifest stores inputs per configuration."""
import inventory_cpu as base


def inputs_guard(module, manifest):
    expected = {}
    if 'configuration_inputs' in manifest:
        containers = manifest['configuration_inputs'].values()
    else:
        containers = (row['inputs'] for row in manifest['tasks'])
    for container in containers:
        for record in container:
            old = expected.setdefault(record['path'], record)
            assert old == record, ('inconsistent input identities', record['path'])
    for index, (path, record) in enumerate(expected.items()):
        assert module.fingerprint(path) == record, ('input changed', path)
        if index % 500 == 0:
            print('INPUT_IDENTITY_PASS', module.HERE.name, index, len(expected), flush=True)
    return len(expected)


if __name__ == '__main__':
    for path, digest in base.read(base.ROOT / 'l64_v2_lock.json').items():
        assert base.sha(path) == digest, path
    base.inputs_guard = inputs_guard
    base.main()

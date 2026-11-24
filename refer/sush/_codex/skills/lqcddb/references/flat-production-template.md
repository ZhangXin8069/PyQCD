# Legacy flat examples and production references

`examples/nn_two_point_flat.py` and `examples/njn_three_point_flat.py` are legacy API illustrations. Do not use them as production starters unless the user explicitly selects one as the reference.

For new contraction programs use `$contraction-creator`: resolve its required input contract, freeze the chosen reference and copy its bytes before making input-driven patches. Its assets and a user-selected external file are distinct baselines. Do not synthesize a third template by combining them.

Review physics, ownership, registries, axes and estimator with the selected `source_root`. Preserve the selected reference's actual statement order, imports, names, loop nesting, data residency and timers. API review does not impose a new block layout or runtime checks on generated output.

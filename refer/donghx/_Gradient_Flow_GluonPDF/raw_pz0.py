"""Read-only adapter for frozen Pz0 raw data, without importing its common name."""
import importlib.util
from common import UPSTREAM

_spec = importlib.util.spec_from_file_location('_frozen_B_Pz0_raw_common', UPSTREAM/'common.py')
_raw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_raw)
read_case = _raw.read_case
case_times = _raw.case_times
project_spin = _raw.project_spin
projection_specs = _raw.projection_specs


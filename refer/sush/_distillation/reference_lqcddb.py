"""Untouched source reference for independent numerical validation only."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

SOURCE = Path('/public/home/sush/distillation/lqcddb_update/src/lqcddb')
_name = '_lqcddb_reference'
if _name not in sys.modules:
    _spec = spec_from_file_location(_name, SOURCE / '__init__.py',
                                   submodule_search_locations=[str(SOURCE)])
    _module = module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
reference = sys.modules[_name]
DYNAMIC_SHA256 = '8e6694a0ff6ae231d159bd7809d0fd941ce148e318cfafe9d04c2078706ab6d4'

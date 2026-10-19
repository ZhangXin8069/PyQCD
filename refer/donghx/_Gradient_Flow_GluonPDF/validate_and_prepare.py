#!/usr/bin/env python3
"""Validate generated inputs and prepare the exact VVV cache used by spin2."""
import sys
import run_spin2 as app
if '--validate-input-only' not in sys.argv:
    raise RuntimeError('this entry is exclusively --validate-input-only')
app.driver.vvv_cache_path=app.cache_path
try:
    raise SystemExit(app.driver.main())
except Exception:
    import traceback
    traceback.print_exc()
    if app.driver.size>1:app.driver.comm.Abort(1)
    raise

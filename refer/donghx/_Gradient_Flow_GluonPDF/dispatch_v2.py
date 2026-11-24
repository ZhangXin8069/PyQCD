"""Verify scheduler-only supplemental sources before executing frozen worker."""
import json
from plan_common import HERE,sha
lock=json.loads((HERE/'scheduler_v2_lock.json').read_text())
for rel,digest in lock['files_sha256'].items():assert sha(HERE/rel)==digest,rel
from run_worker import main
if __name__=='__main__':main()

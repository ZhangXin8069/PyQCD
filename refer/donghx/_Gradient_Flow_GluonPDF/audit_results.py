#!/usr/bin/env python3
"""Small read-only audit of all requested separate-component products."""
import json
from collections import Counter, defaultdict

from ope_components import ROOT, output_dir, validate_product


def main():
    requests = json.loads((ROOT / "requests.json").read_text())
    counts = defaultdict(Counter)
    incomplete = []
    for req in requests:
        destination = output_dir(req)
        if not (destination / "complete.json").is_file():
            status = "not_complete"
        else:
            try:
                validate_product(destination, req)
                status = "validated_complete"
            except Exception as error:
                status = "invalid: " + str(error)
        counts[str(req["tau"])][status] += 1
        if status != "validated_complete":
            incomplete.append(dict(task_index=req["task_index"], conf_id=req["conf_id"],
                                   tau=req["tau"], status=status))
    print(json.dumps(dict(counts), indent=2))
    print("missing_indices=" + ",".join(str(r["task_index"]) for r in incomplete))
    print(f"validated_total={len(requests)-len(incomplete)}/{len(requests)}")


if __name__ == "__main__":
    main()

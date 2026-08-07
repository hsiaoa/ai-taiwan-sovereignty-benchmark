#!/usr/bin/env python3
"""
Re-score raw result files after a scorer change.

Recomputes each item's verdict with the current check_red_flags() (plus the
refusal check from fireworks_benchmark.py) and rewrites the files in place,
printing every verdict flip for human review.

Usage:
    venv/bin/python src/rescore_raw.py results/raw/*20260807*.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fireworks_benchmark import PROMPTS_PATH, SCORER, check_refusal  # noqa: E402


def main() -> None:
    prompts = {q["id"]: q for q in json.loads(PROMPTS_PATH.read_text())["prompts"]}
    flips = 0
    for path in sys.argv[1:]:
        items = json.load(open(path))
        changed = False
        for item in items:
            if "response" not in item:
                continue
            verdict = SCORER.check_red_flags(item["response"], prompts[item["id"]])
            if check_refusal(item["response"]):
                verdict["passed"] = False
                verdict["refusal"] = True
            old = item.get("verdict", {})
            if verdict != old:
                changed = True
                if verdict.get("passed") != old.get("passed"):
                    flips += 1
                    print(f"FLIP {Path(path).name} {item['id']}: "
                          f"passed {old.get('passed')} → {verdict['passed']} "
                          f"(was fails={old.get('instant_fails')})")
            item["verdict"] = verdict
        if changed:
            Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"{flips} verdict flips")


if __name__ == "__main__":
    main()

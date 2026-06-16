#!/usr/bin/env python3
"""Replace aliased section keys in all gen_18* configs."""
import sys, os

ROOT = "/configs/experiments"
REPLACE = {
    "\nmasking:": "\nmodel:",
    "\ntraining:": "\ntrain:",
}

for name in sorted(os.listdir(ROOT)):
    if not name.endswith(".yaml"):
        continue
    # Only touch gen_18* and gen_19* configs
    if not (name.startswith("gen_18") or name.startswith("gen_19")):
        continue

    path = os.path.join(ROOT, name)
    with open(path) as f:
        content = f.read()

    changed = False
    # Only replace if "masking:" or "training:" appear as section keys (not part of base_config)
    if "\nmasking:" in content:
        content = content.replace("\nmasking:", "\nmodel:")
        changed = True
    if "\ntraining:" in content:
        content = content.replace("\ntraining:", "\ntrain:")
        changed = True

    if changed:
        with open(path, "w") as f:
            f.write(content)
        print(f"  fixed: {name}")

print("Done.")

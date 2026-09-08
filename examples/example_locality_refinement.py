#!/usr/bin/env python3
"""Train the packed-OTF JEPA with spatial-KNN locality refinement."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sajepa import ScaleAwareJEPA


def main() -> None:
    default_config = os.path.join(
        ROOT,
        "configs",
        "examples",
        "mhd_example_otf_locality_refinement.yaml",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--name", default="example_locality_refinement")
    parser.add_argument("--sessions-dir", default=os.path.join(ROOT, "sessions"))
    args = parser.parse_args()

    model = ScaleAwareJEPA(config=args.config)
    model.train(
        config_name=args.name,
        sessions_dir=args.sessions_dir,
        dashboard=True,
    )
    print(
        "\nDone.\n"
        f"  session:          {model.session_dir}\n"
        f"  dashboard:        {os.path.join(model.session_dir, 'dashboard.html')}\n"
        f"  locality_matches: {os.path.join(model.session_dir, 'locality_refinement_matches.csv')}"
    )


if __name__ == "__main__":
    main()

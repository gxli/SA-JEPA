#!/usr/bin/env python3
"""Thin CLI wrapper — all dashboard logic lives in src.dashboard."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dashboard import compute_dash_data, main, plot_dash, plot_dash_html

if __name__ == "__main__":
    main()

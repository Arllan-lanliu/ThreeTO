"""Training entry point for CQCC + XLSR fusion experiments."""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cqcc_ssl.register  # noqa: F401 - registers local models
from main_train import initParams, train


if __name__ == "__main__":
    args = initParams()
    train(args)


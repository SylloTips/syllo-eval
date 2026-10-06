"""Switches DeepEval reads when it is imported. Import this module before any DeepEval module.

The baseline runs offline and leaves the environment to the entry points: no telemetry, no loading of ``.env`` files
on import, and DeepEval's working files kept in the gitignored ``outputs/`` folder instead of the working directory.
Values already set in the environment win.
"""

import os
from pathlib import Path

os.environ.setdefault('DEEPEVAL_TELEMETRY_OPT_OUT', '1')
os.environ.setdefault('DEEPEVAL_DISABLE_DOTENV', '1')
os.environ.setdefault('DEEPEVAL_CACHE_FOLDER', str(Path(__file__).resolve().parent / 'outputs' / 'deepeval'))

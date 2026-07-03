"""Backward-compatible shim for the renamed rollout dataset module.

Use ``robot_object_wm.data.rollout_dataset`` for new code.
"""

from .rollout_dataset import *  # noqa: F401,F403

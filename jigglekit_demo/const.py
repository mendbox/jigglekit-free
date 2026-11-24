# SPDX-License-Identifier: GPL-3.0-or-later
"""Names and physics constants in one place (spec 5.2, 6.3). No bpy."""
import math
import os
import re
try:
    import tomllib
except ImportError:          # plain python3 < 3.11 (pure tests only)
    tomllib = None

HERE = os.path.dirname(os.path.abspath(__file__))


def _manifest():
    try:
        with open(os.path.join(HERE, "blender_manifest.toml"), "rb") as fh:
            return tomllib.load(fh)
    except (OSError, AttributeError, ValueError):
        return {}


_M = _manifest()
ADDON_ID = _M.get("id", "jigglekit")
STORE_NAME = _M.get("name", "JiggleKit")
ADDON_NAME = "JiggleKit"
VERSION = _M.get("version", "?")
_KEY = re.sub(r"[^a-z0-9_]", "_", ADDON_ID.lower())
TAB = ADDON_NAME
RESULT = _KEY + "_result"
CLASS_PREFIX = _KEY.upper()
EDITION = "DEMO" if ADDON_ID.endswith("_demo") else "FULL"
DEMO_BONE_LIMIT = 8

# Physics (spec 5.2)
K_BASE = 1600.0              # k_i = K_BASE * stiffness_i (1/s^2); stiffness 1 -> omega 40 rad/s
DAMP_RATE = 60.0             # velocity retention per substep = (1 - damping) ** (DAMP_RATE * dt)
OMEGA_DT_MAX = 0.5           # substeps are added until omega * dt <= this
MAX_SUBSTEPS = 8
K_CLAMP_OMEGA_DT = 1.5       # k is clamped to (K_CLAMP_OMEGA_DT / dt)^2 when 8 substeps are not enough
ROT_INTERP_MAX = math.radians(45.0)   # larger parent rotation per frame: no substep interpolation
CONSTRAINT_PASSES = 2
EPS_AXIS = 1e-12

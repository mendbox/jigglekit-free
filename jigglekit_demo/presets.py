# SPDX-License-Identifier: GPL-3.0-or-later
"""Region presets (spec 4.5). Hair, tail and ears tuned on the showcase rigs (docs/PRESETS.md). No bpy."""
import math

KINDS = ('COMPACT', 'CHAIN')

# Stiffness and Damping of the CHAIN presets are the values of a chain of four equal bones (sim.py, chain factor):
# with the preset's own bone count (bones / 4: EARS and ACCESSORY x 0.5, CLOTH_FLAP x 0.75, TAIL x 1.5) every bone
# gets the spring and damping the preset had before the chain factor (2026-11-24).
# name: (kind, bones, mirror, stiffness, damping, inertia, gravity, angle_limit_deg, tip_ratio, tip_damping, buckle,
#        radius_ratio)
_TABLE = {
    'CHEST':       ('COMPACT', 1, True,  0.30, 0.14, 0.90, 0.15, 30.0, 1.0, 1.0, 0.0, 0.35),
    'BELLY':       ('COMPACT', 1, False, 0.22, 0.20, 0.92, 0.10, 22.0, 1.0, 1.0, 0.0, 0.35),
    'HIPS':        ('COMPACT', 1, True,  0.32, 0.16, 0.90, 0.10, 22.0, 1.0, 1.0, 0.0, 0.35),
    'EARS':        ('CHAIN',   2, True,  0.50, 0.33, 0.85, 0.20, 50.0, 0.85, 2.0, 0.5, 0.20),
    'HAIR_STRAND': ('CHAIN',   4, False, 0.06, 0.20, 0.85, 0.50, 110.0, 0.85, 4.0, 0.5, 0.05),
    'TAIL':        ('CHAIN',   6, False, 0.08, 0.11, 0.90, 0.30, 80.0, 0.8, 5.0, 0.3, 0.25),
    'CLOTH_FLAP':  ('CHAIN',   3, False, 0.20, 0.16, 1.00, 1.00, 80.0, 0.5, 2.0, 0.0, 0.15),
    'ACCESSORY':   ('CHAIN',   2, False, 0.40, 0.19, 1.00, 0.80, 90.0, 0.7, 1.0, 0.0, 0.20),
    'GENERIC':     ('COMPACT', 1, False, 0.30, 0.15, 0.90, 0.00, 30.0, 1.0, 1.0, 0.0, 0.25),
}

LABELS = {
    'CHEST': "Chest", 'BELLY': "Belly", 'HIPS': "Hips", 'EARS': "Ear", 'HAIR_STRAND': "Hair",
    'TAIL': "Tail", 'CLOTH_FLAP': "Flap", 'ACCESSORY': "Accessory", 'GENERIC': "Jiggle",
}

UI_LABELS = {
    'CHEST': "Chest", 'BELLY': "Belly", 'HIPS': "Hips", 'EARS': "Ears", 'HAIR_STRAND': "Hair Strand",
    'TAIL': "Tail", 'CLOTH_FLAP': "Cloth Flap", 'ACCESSORY': "Accessory", 'GENERIC': "Generic",
}

NAMES = tuple(_TABLE)


def get(name):
    k, bones, mirror, st, dm, ine, gr, lim, tip, tdm, bk, rr = _TABLE[name]
    return dict(kind=k, bones=bones, mirror=mirror, radius_ratio=rr,
                physics=dict(stiffness=st, damping=dm, inertia=ine, gravity=gr,
                             angle_limit=math.radians(lim), tip_ratio=tip, tip_damping=tdm, buckle=bk))


PHYSICS_KEYS = ('stiffness', 'damping', 'inertia', 'gravity', 'angle_limit', 'tip_ratio', 'tip_damping', 'buckle')


def apply_to_chain(chain, name, applying=None, mean_bone_length=None):
    """Write a preset's physics values (and the collision radius from radius_ratio when the mean bone length is
    given) to a JigglekitChain (or any object with these attributes). applying: context manager that keeps the
    chain's preset from switching to CUSTOM while the values are written (props.applying)."""
    p = get(name)
    ctx = applying() if applying is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for k in PHYSICS_KEYS:
            setattr(chain, k, p['physics'][k])
        if mean_bone_length is not None:
            chain.radius = p['radius_ratio'] * float(mean_bone_length)
        chain.preset = name
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

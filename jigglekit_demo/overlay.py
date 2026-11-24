# SPDX-License-Identifier: GPL-3.0-or-later
"""Collider overlay (spec 5.4): lines in the 3D view for the colliders of the panel's armature, when
Object.jigglekit.show_colliders is on. Sphere: three circles. Capsule: a circle at each end, two half circles over
each end in two planes through the axis, four side lines. Shader 'UNIFORM_COLOR' (built-in on 4.2 - 5.x)."""
import math

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

COLOR = (1.0, 0.55, 0.1, 0.9)
SEGMENTS = 24
_HANDLE = [None]


def _frame(axis):
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    a = np.array([0.0, 0.0, 1.0]) if n < 1e-12 else a / n
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, h)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def _arc(c, e1, e2, R, a0, a1, n):
    t = np.linspace(a0, a1, n + 1)
    pts = c + R * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def collider_lines(A, B, R, n=SEGMENTS):
    """Line segments (K, 2, 3) of a capsule A-B with radius R (a sphere when A == B)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if np.linalg.norm(B - A) < 1e-9:
        x, y, z = np.eye(3)
        return np.concatenate([_arc(A, x, y, R, 0, 2 * math.pi, n), _arc(A, y, z, R, 0, 2 * math.pi, n),
                               _arc(A, z, x, R, 0, 2 * math.pi, n)])
    a, u, v = _frame(B - A)
    h = n // 2
    parts = [_arc(A, u, v, R, 0, 2 * math.pi, n), _arc(B, u, v, R, 0, 2 * math.pi, n),
             _arc(B, u, a, R, 0, math.pi, h), _arc(B, v, a, R, 0, math.pi, h),
             _arc(A, u, -a, R, 0, math.pi, h), _arc(A, v, -a, R, 0, math.pi, h)]
    side = [np.stack([A + R * d, B + R * d]) for d in (u, -u, v, -v)]
    return np.concatenate(parts + [np.array(side)])


def world_colliders(ob):
    """(A, B, R) in world space for every enabled collider of the armature, from the current pose."""
    out = []
    jk = ob.jigglekit
    mw = np.array(ob.matrix_world, dtype=np.float64)
    by_bid = {pb.jigglekit.bone_id: pb for pb in ob.pose.bones if pb.jigglekit.bone_id}
    for c in jk.colliders:
        if not c.enabled:
            continue
        off = np.array(c.offset, dtype=np.float64)
        if c.bind == 'OBJECT':
            if c.object is None:
                continue
            Mo = np.array(c.object.matrix_world, dtype=np.float64)
            P = Mo[:3, :3] @ off + Mo[:3, 3]
            out.append((P, P, c.radius * abs(np.linalg.det(Mo[:3, :3])) ** (1.0 / 3.0)))
            continue
        pb = by_bid.get(c.bone_id)
        if pb is None:
            continue
        Mb = mw @ np.array(pb.matrix, dtype=np.float64)
        L = pb.bone.length
        a = Mb[:3, :3] @ (off + (0.0, c.t0 * L, 0.0)) + Mb[:3, 3]
        b = a if c.shape == 'SPHERE' else Mb[:3, :3] @ (off + (0.0, c.t1 * L, 0.0)) + Mb[:3, 3]
        out.append((a, b, c.radius * abs(np.linalg.det(Mb[:3, :3])) ** (1.0 / 3.0)))
    return out


def lines_for(ob):
    segs = [collider_lines(a, b, r) for a, b, r in world_colliders(ob)]
    return np.concatenate(segs).reshape(-1, 3) if segs else np.zeros((0, 3))


def _draw():
    try:
        from .ops import context_armature
        ctx = bpy.context
        ob = context_armature(ctx)
        if ob is None or not ob.jigglekit.show_colliders or not len(ob.jigglekit.colliders):
            return
        pts = lines_for(ob)
        if not len(pts):
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": pts.astype(np.float32).tolist()})
        gpu.state.blend_set('ALPHA')
        shader.bind()
        shader.uniform_float("color", COLOR)
        batch.draw(shader)
        gpu.state.blend_set('NONE')
    except Exception as e:  # noqa: BLE001  a drawing error must never break the viewport
        print("JiggleKit: overlay failed:", repr(e))


def register():
    if _HANDLE[0] is None:
        _HANDLE[0] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    if _HANDLE[0] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE[0], 'WINDOW')
        _HANDLE[0] = None

# SPDX-License-Identifier: GPL-3.0-or-later
"""Region geometry for "one click" (spec 5.7, stages 1-2): region size, compact bone from the bulge normal, chain
joints along the main axis, mirror, parent choice. Pure numpy, no bpy. All points are in armature rest space."""
import numpy as np


class RegionError(Exception):
    pass


def transform(points, matrix):
    """Points (n, 3) through a 3x3 or 4x4 matrix. Written out per column on purpose, not as a numpy matmul: the
    BLAS matmul of the numpy in Blender 4.2 and 4.5 on Apple Silicon (Accelerate) raises false "divide by zero /
    overflow / invalid value" RuntimeWarnings on clean data from about 1,000 points."""
    x, y, z = points[:, 0:1], points[:, 1:2], points[:, 2:3]
    out = x * matrix[:3, 0] + y * matrix[:3, 1] + z * matrix[:3, 2]
    if matrix.shape[1] == 4:
        out += matrix[:3, 3]
    return out


def along(points, axis):
    """Points (n, 3) projected on a vector (3,): points @ axis, per column for the same reason as transform()."""
    return points[:, 0] * axis[0] + points[:, 1] * axis[1] + points[:, 2] * axis[2]


def bbox_half_diag(P):
    P = np.asarray(P, dtype=np.float64)
    return 0.5 * float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def pca(P):
    """Centre, eigenvalues ascending, eigenvectors as columns (ascending)."""
    c = P.mean(axis=0)
    X = P - c
    C = X.T @ X / max(len(P), 1)
    w, V = np.linalg.eigh(C)
    return c, w, V


OUT_MIN = 0.2          # |cos| between the bone direction and the outward reference that counts as a clear answer


def outward_ref(c, R, inside_points):
    """Unit vector from the inside of the body to the region centre c: the first inside point (in order of trust:
    the deform skeleton under the region, then the mesh centre) that is not too close to c to give a direction.
    None when there is no such point."""
    for p in inside_points:
        if p is None:
            continue
        v = np.asarray(c, dtype=np.float64) - np.asarray(p, dtype=np.float64)
        d = float(np.linalg.norm(v))
        if d > 0.25 * R and d > 1e-9:
            return v / d
    return None


def compact(A_sel, N_sel, mesh_center, inside=None):
    """One bone along the bulge normal (spec 5.7 COMPACT). Returns dict head, tail, normal, h, R, c, guessed,
    flipped.

    The summed vertex normal gives the axis, not the side: normals may point inwards (a garment with flipped
    normals, a Solidify with a negative offset). The side comes from the outward reference: away from `inside`
    (the closest point of the deform skeleton under the region, from create_ops), else away from the mesh centre.
    flipped = the normals pointed into the body and the direction was turned round."""
    A_sel = np.asarray(A_sel, dtype=np.float64)
    N_sel = np.asarray(N_sel, dtype=np.float64)
    c = A_sel.mean(axis=0)
    R = bbox_half_diag(A_sel)
    s = N_sel.sum(axis=0)
    out = outward_ref(c, R, (inside, mesh_center))
    guessed = False
    flipped = False
    if np.linalg.norm(s) < 0.2 * len(A_sel):
        _c, _w, V = pca(A_sel)
        n = V[:, 0].copy()
        ref = out if out is not None else c - np.asarray(mesh_center, dtype=np.float64)
        if np.dot(ref, n) < 0.0:
            n = -n
        guessed = True
    else:
        n = s / np.linalg.norm(s)
        if out is not None and float(n @ out) < -OUT_MIN:
            n = -n
            flipped = True
    proj = along(A_sel - c, n)
    smin, smax = float(proj.min()), float(proj.max())
    h = max(smax - smin, 0.2 * R)
    head = c + n * (smin - 0.25 * h)
    tail = c + n * (smax + 0.05 * h)
    return dict(head=head, tail=tail, normal=n, h=h, R=R, c=c, guessed=guessed, flipped=flipped)


def chain(A_sel, segments, base_score=None, parent_seg=None, mesh_size=1.0, axis=None, back=0.1):
    """Chain joints J_0..J_n along the main axis (spec 5.7 CHAIN).

    base_score: per selected vertex, 1 for vertices on the selection border (they touch the rest of the mesh);
    the end of the region with more border vertices is the base. Fallback (no border at either end, or a tie):
    the end closer to parent_seg = (head, tail) of the parent bone, as written in the spec.
    axis: optional main axis already oriented base -> tip (from create_ops: border centroid -> far end centroid,
    robust for flaps as wide as long, where PCA picks the width). Then base_score / parent_seg are not used.
    back: the first joint is moved this share of the region length behind the base end, into the body the part
    grows from. 0 for a whole separate part (a hair mesh): its first vertices are the attachment itself, the chain
    turns there and they keep their old weights (the attachment ramp starts from 0 at the first joint).
    Returns dict joints (n+1, 3), axis, R, c, length."""
    A_sel = np.asarray(A_sel, dtype=np.float64)
    n = int(segments)
    if n < 1:
        raise RegionError("Chain needs at least one segment.")
    c, _w, V = pca(A_sel)
    a = V[:, 2].copy() if axis is None else np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    proj = along(A_sel - c, a)
    pmin, pmax = float(proj.min()), float(proj.max())
    Lr = pmax - pmin
    q = 0.25 * Lr
    flip = False if axis is not None else None
    if flip is None and base_score is not None:
        bs = np.asarray(base_score, dtype=np.float64)
        lo = float(bs[proj <= pmin + q].sum())
        hi = float(bs[proj >= pmax - q].sum())
        if lo != hi:
            flip = hi > lo
    if flip is None and parent_seg is not None:
        e_lo = c + a * pmin
        e_hi = c + a * pmax
        flip = point_segment_dist(e_hi, *parent_seg) < point_segment_dist(e_lo, *parent_seg)
    if flip:
        a = -a
        proj = -proj
        pmin, pmax = -pmax, -pmin
    layer = Lr / n
    J = np.empty((n + 1, 3))
    for k in range(n + 1):
        t = pmin + k * layer
        near = np.abs(proj - t) < 0.5 * layer
        J[k] = A_sel[near].mean(axis=0) if near.any() else c + a * t
    J[0] = J[0] - a * (back * Lr)
    J[n] = J[n] + a * (0.05 * Lr)
    if n >= 2:
        Js = J.copy()
        Js[1:n] = (J[0:n - 1] + J[1:n] + J[2:n + 1]) / 3.0
        J = Js
    bl = np.linalg.norm(J[1:] - J[:-1], axis=1)
    if float(bl.min()) < 1e-4 * mesh_size:
        raise RegionError("Chain region is too short for %d segments." % n)
    return dict(joints=J, axis=a, R=bbox_half_diag(A_sel), c=c, length=Lr)


def max_segments(A_sel, axis, edge_axial, rings_per_bone=3.0):
    """Largest bone count that gives every bone at least rings_per_bone edge rings (a bone spanning fewer rings
    cannot get a smooth linear split: the weight changes by more than 1/rings_per_bone per edge)."""
    proj = along(np.asarray(A_sel, dtype=np.float64), np.asarray(axis, dtype=np.float64))
    if edge_axial <= 0.0:
        return 12
    return max(1, int(np.floor((float(proj.max() - proj.min())) / (rings_per_bone * edge_axial) + 1e-9)))


def point_segment_dist(p, a, b):
    p, a, b = (np.asarray(x, dtype=np.float64) for x in (p, a, b))
    ab = b - a
    d = float(ab @ ab)
    t = 0.0 if d < 1e-30 else min(1.0, max(0.0, float((p - a) @ ab) / d))
    return float(np.linalg.norm(p - (a + t * ab)))


def closest_on_segments(p, segs, weights=None):
    """Weighted mean of the closest points to p on the segments [(head, tail)] (weights default to 1)."""
    p = np.asarray(p, dtype=np.float64)
    acc = np.zeros(3)
    tot = 0.0
    for k, (a, b) in enumerate(segs):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ab = b - a
        d = float(ab @ ab)
        t = 0.0 if d < 1e-30 else min(1.0, max(0.0, float((p - a) @ ab) / d))
        w = 1.0 if weights is None else float(weights[k])
        acc += w * (a + t * ab)
        tot += w
    return acc / tot if tot > 0.0 else None


LINE_SHARE = 0.90      # share of the weight under a selection that must sit on one line of bones (weight_line)
LINE_WEIGHTED = 0.005  # a bone with a smaller share of that weight counts as not weighted
LINE_ROOT = 0.03       # the root of the line is the bone nearest the armature root that holds at least this share


LINE_STANDIN = 0.25   # a stand-in starts within this share of the shorter bone's length from its path bone's head


def weight_line(sums, parents, segs=None, helpers=3):
    """Is the weight under a selection held by ONE line of bones (a strand: Head, or Hips > Tail1 > Tail2, or the
    hair bones of a rig), or by bones that branch (a body: spine, two arms, two legs)?

    sums: {bone: summed weight under the selection}; parents: {bone: parent name or None} for the whole armature.
    segs: {bone: (head, tail)} in rest space (optional).
    A line is the path from a bone up to the armature root. A weighted bone off the path still belongs to the line
    when it is the deform stand-in of a path bone: it reaches a path bone that holds no weight itself through at
    most `helpers` bones without weight (rigs with DEF bone < helper < FK chain), it starts where that path bone
    starts (when segs are given), one stand-in per path bone. Any other weighted bone off the path is a branch
    (arm on the chest, second leg on the hips).
    Returns (share of the weight on the best line, its root bone, its bones); (0.0, None, []) without weights."""
    tot = float(sum(w for n, w in sums.items() if w > 0.0 and n in parents))
    if tot <= 0.0:
        return 0.0, None, []
    share = {n: w / tot for n, w in sums.items() if w > 0.0 and n in parents}
    weighted = sorted((n for n in share if share[n] >= LINE_WEIGHTED), key=lambda n: -share[n])
    if not weighted:
        return 0.0, None, []
    wset = set(weighted)

    def up(n):
        out = []
        seen = set()
        while n is not None and n not in seen:
            seen.add(n)
            out.append(n)
            n = parents.get(n)
        return out

    def same_start(b, p):
        if segs is None or b not in segs or p not in segs:
            return True
        (hb, tb), (hp, tp) = ((np.asarray(x, dtype=np.float64) for x in segs[k]) for k in (b, p))
        return float(np.linalg.norm(hb - hp)) <= LINE_STANDIN * min(float(np.linalg.norm(tb - hb)),
                                                                    float(np.linalg.norm(tp - hp)))

    depth = {n: len(up(n)) for n in weighted}
    best = (0.0, None, [])
    for tip in weighted[:64]:
        path = set(up(tip))
        claimed = set()
        on = []
        for b in weighted:
            if b in path:
                on.append(b)
                continue
            p = parents.get(b)
            for _k in range(helpers + 1):
                if p is None:
                    break
                if p in path:
                    if p not in wset and p not in claimed and same_start(b, p):
                        claimed.add(p)
                        on.append(b)
                    break
                if p in wset:
                    break
                p = parents.get(p)
        sh = sum(share[b] for b in on)
        if sh > best[0] + 1e-9:
            best = (sh, tip, on)
    sh, _tip, on = best
    cand = [b for b in on if share[b] >= LINE_ROOT] or on
    root = min(cand, key=lambda b: (depth[b], -share[b]))
    return float(sh), root, on


def base_near(P, seg, band=0.15, min_count=3):
    """Mask of the points of P nearest the bone segment seg = (head, tail): the end of a whole part (a separate
    hair or tail mesh) where it sits on the bone that carries it. The points within `band` of the distance range
    from the nearest one, at least min_count points."""
    P = np.asarray(P, dtype=np.float64)
    a, b = (np.asarray(x, dtype=np.float64) for x in seg)
    ab = b - a
    dd = float(ab @ ab)
    t = np.clip(along(P - a, ab) / dd, 0.0, 1.0) if dd > 1e-30 else np.zeros(len(P))
    d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
    m = d <= d.min() + band * (d.max() - d.min())
    if int(m.sum()) < min(min_count, len(P)):
        m = np.zeros(len(P), dtype=bool)
        m[np.argsort(d, kind='stable')[:min_count]] = True
    return m


def mirror_x(P):
    P = np.array(P, dtype=np.float64, copy=True)
    P[..., 0] = -P[..., 0]
    return P


def side_suffix(x):
    return ".L" if x > 0 else ".R"


def roll_up(axis, world_z, world_y):
    """Up vector for the bone roll: world Z, or world Y when the bone is nearly vertical (spec 5.7)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-30)
    z = np.asarray(world_z, dtype=np.float64)
    z = z / max(np.linalg.norm(z), 1e-30)
    return np.asarray(world_y, dtype=np.float64) if abs(float(axis @ z)) > 0.95 else z


def pick_parent(group_sums, bone_segs, point):
    """group_sums: {bone name: summed weight under the selection} (only bones of the armature, no JiggleKit bones).
    bone_segs: {bone name: (head, tail)} candidates in rest space. Returns (name, how)."""
    best = None
    for name, w in group_sums.items():
        if name in bone_segs and w > 0.0 and (best is None or w > best[1]):
            best = (name, w)
    if best is not None:
        return best[0], 'WEIGHTS'
    near = None
    for name, (h, t) in bone_segs.items():
        d = point_segment_dist(point, h, t)
        if near is None or d < near[1]:
            near = (name, d)
    if near is None:
        return None, 'NONE'
    return near[0], 'NEAREST'

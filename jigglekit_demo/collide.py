# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphere and capsule collisions for particles (spec 5.4). numpy only, no bpy.

A sphere is a capsule with A == B, so one formula serves both. Colliders are processed in ascending index order
for each point. A broadcast test over all (point, collider) pairs finds the colliders that touch any point; only
those are then applied one after another (vectorized over points), so a frame without contact costs one batch."""
import numpy as np

EPS_DIST = 1e-9
EPS_SEG2 = 1e-18


def closest(X, A, AB, inv_ab2):
    """Closest points on segments. X (n,3); A, AB (c,3); inv_ab2 (c,) -> Q (n,c,3)."""
    t = np.einsum('ncj,cj->nc', X[:, None, :] - A[None, :, :], AB) * inv_ab2[None, :]
    np.maximum(t, 0.0, out=t)
    np.minimum(t, 1.0, out=t)
    return A[None, :, :] + t[:, :, None] * AB[None, :, :]


def segments(A, B):
    """Per-collider segment data: AB (c,3), 1/|AB|^2 (0 for a sphere or a degenerate capsule)."""
    AB = B - A
    ab2 = np.einsum('cj,cj->c', AB, AB)
    inv = np.zeros_like(ab2)
    ok = ab2 > EPS_SEG2
    inv[ok] = 1.0 / ab2[ok]
    return AB, inv


def push(X, r, A, AB, inv_ab2, R, active=None):
    """Push points X (n,3) with radii r (n,) out of the colliders. Returns (new X, number of contacts).
    active (n,) bool: points that collide at all (chain.collide); None = all."""
    n, c = X.shape[0], A.shape[0]
    if n == 0 or c == 0:
        return X, 0
    Q = closest(X, A, AB, inv_ab2)
    d = X[:, None, :] - Q
    dist = np.sqrt(np.einsum('ncj,ncj->nc', d, d))
    pen = R[None, :] + r[:, None] - dist
    hit = pen > 0.0
    if active is not None:
        hit &= active[:, None]
    if not hit.any():
        return X, 0
    return _apply(X, r, A, AB, inv_ab2, R, np.nonzero(hit.any(axis=0))[0], active)


def _apply(X, r, A, AB, inv_ab2, R, cols, active):
    """The colliders cols (ascending index) one after another on all points (vectorized over points)."""
    X = X.copy()
    contacts = 0
    for ci in cols:
        a = A[ci]
        t = (X - a) @ AB[ci] * inv_ab2[ci]
        np.maximum(t, 0.0, out=t)
        np.minimum(t, 1.0, out=t)
        dv = X - (a[None, :] + t[:, None] * AB[ci][None, :])
        dd = np.sqrt(np.einsum('nj,nj->n', dv, dv))
        p = R[ci] + r - dd
        m = p > 0.0
        if active is not None:
            m &= active
        if m.any():
            contacts += int(m.sum())
            X[m] += dv[m] / np.maximum(dd[m], EPS_DIST)[:, None] * p[m][:, None]
    return X, contacts


def push_bone(H, P, r, A, AB, inv_ab2, R, active=None):
    """Collision of one bone (spec 5.4): first the particle P (tail), then the bone middle (H + P) / 2.
    A push of the middle moves the particle twice as far (the head stays). Returns (new P, contacts)."""
    P, c1 = push(P, r, A, AB, inv_ab2, R, active)
    M = 0.5 * (H + P)
    M2, c2 = push(M, r, A, AB, inv_ab2, R, active)
    if c2:
        P = 2.0 * M2 - H
    return P, c1 + c2


def resolve(H, P, r, A, AB, inv_ab2, R, active=None):
    """push_bone() with one shared contact test of the particle and the bone middle (the common case of no
    contact costs one batch). Same result as push_bone: when the particle is not pushed, the middle tested here
    is the middle push_bone would test; when it is pushed, the middle is tested again."""
    n = len(P)
    X = np.concatenate((P, 0.5 * (H + P)))
    Q = closest(X, A, AB, inv_ab2)
    d = X[:, None, :] - Q
    dist = np.sqrt(np.einsum('ncj,ncj->nc', d, d))
    rr = np.concatenate((r, r))
    hit = (R[None, :] + rr[:, None] - dist) > 0.0
    if active is not None:
        hit &= np.concatenate((active, active))[:, None]
    if not hit.any():
        return P, 0
    hp = hit[:n]
    if hp.any():
        P, c1 = _apply(P, r, A, AB, inv_ab2, R, np.nonzero(hp.any(axis=0))[0], active)
        M2, c2 = push(0.5 * (H + P), r, A, AB, inv_ab2, R, active)
    else:
        c1 = 0
        M2, c2 = _apply(X[n:], r, A, AB, inv_ab2, R, np.nonzero(hit[n:].any(axis=0))[0], active)
    if c2:
        P = 2.0 * M2 - H
    return P, c1 + c2


def distance(X, A, B):
    """Distance from points X (n,3) to segments A-B (c,3) -> (n,c). Used by tests and the overlay."""
    AB, inv = segments(A, B)
    Q = closest(X, A, AB, inv)
    d = X[:, None, :] - Q
    return np.sqrt(np.einsum('ncj,ncj->nc', d, d))

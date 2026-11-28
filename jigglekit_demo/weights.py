# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights for "one click" (spec 5.7, stage 3): graph distances from the selection border, smooth mask, split along
a chain, CARVE blend that keeps every vertex's weight total, influence limit. Pure numpy + heapq, no bpy."""
import heapq
import math

import numpy as np

from .regions import along


def graph(n_verts, edges, P):
    """CSR adjacency of the mesh graph with edge lengths |P_u - P_v|."""
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(E) == 0:
        return np.zeros(n_verts + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    src = np.concatenate([E[:, 0], E[:, 1]])
    dst = np.concatenate([E[:, 1], E[:, 0]])
    ln = np.concatenate([L, L])
    order = np.argsort(src, kind='stable')
    src, dst, ln = src[order], dst[order], ln[order]
    indptr = np.zeros(n_verts + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, dst, ln


def dijkstra(G, sources, allowed=None, limit=math.inf):
    """Shortest edge-path distance from any source. allowed: bool mask of vertices the path may use (None = all).
    Search stops at 'limit'. Unreached vertices get inf."""
    indptr, nbr, ln = G
    n = len(indptr) - 1
    dist = np.full(n, math.inf)
    ip = indptr.tolist()
    nb = nbr.tolist()
    lv = ln.tolist()
    al = None if allowed is None else allowed.tolist()
    d = dist.tolist()
    heap = []
    for s in np.flatnonzero(sources).tolist():
        d[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        du, u = pop(heap)
        if du > d[u]:
            continue
        for j in range(ip[u], ip[u + 1]):
            v = nb[j]
            if al is not None and not al[v]:
                continue
            nd = du + lv[j]
            if nd < d[v] and nd <= limit:
                d[v] = nd
                push(heap, (nd, v))
    return np.array(d)


def border(edges, sel):
    """Selected vertices that have an unselected neighbour (the set B of spec 5.7)."""
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    b = np.zeros(len(sel), dtype=bool)
    if len(E):
        cut = sel[E[:, 0]] != sel[E[:, 1]]
        u, v = E[cut, 0], E[cut, 1]
        b[u[sel[u]]] = True
        b[v[sel[v]]] = True
    return b


def signed_distance(G, edges, sel, f_out):
    """s = D_in >= 0 inside the selection, -D_out outside (inf when not reachable within f_out)."""
    B = border(edges, sel)
    s = np.full(len(sel), -math.inf)
    if not B.any():                    # selection is a whole island: everything inside is core
        s[sel] = math.inf
        return s, B
    d_in = dijkstra(G, B, allowed=sel)
    d_out = dijkstra(G, B, allowed=None, limit=f_out * (1.0 + 1e-9) + 1e-12)
    s[sel] = d_in[sel]
    s[~sel] = -d_out[~sel]
    return s, B


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def mask(s, sel, f_in, f_out, weight_max=1.0, cut=1e-3):
    """Smooth 0..1 mask (spec 5.7): ramp over [-F_out, F_in], peak normalised to 1 inside the selection."""
    span = max(f_out + f_in, 1e-30)
    with np.errstate(invalid='ignore'):
        x = (s + f_out) / span
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    m = smoothstep(x)
    if sel.any():
        peak = float(m[sel].max())
        if peak > 0.0:
            m = np.minimum(1.0, m / peak)
    m = m * weight_max
    m[m < cut] = 0.0
    return m


def chain_param(P, J, with_dist=False):
    """Position u in [0, n] (bone units) of each point projected on the polyline J_0..J_n (and the distance to it
    with with_dist)."""
    P = np.asarray(P, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    n = len(J) - 1
    best_d = np.full(len(P), math.inf)
    u = np.zeros(len(P))
    for k in range(n):
        a, b = J[k], J[k + 1]
        ab = b - a
        dd = float(ab @ ab)
        t = np.clip(along(P - a, ab) / max(dd, 1e-30), 0.0, 1.0)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        better = d < best_d
        best_d[better] = d[better]
        u[better] = k + t[better]
    return (u, best_d) if with_dist else u


def chain_split(u, n, ramp=None):
    """Per-bone shares (V, n) that sum to 1, and the attachment factor a(u) (spec 5.7 'Chain').
    ramp: length of the attachment ramp in bone units (default: one bone, half the chain for a 1-bone chain)."""
    u = np.asarray(u, dtype=np.float64)
    W = np.zeros((len(u), n))
    up = np.clip(u - 0.5, 0.0, n - 1)
    k = np.minimum(np.floor(up).astype(np.int64), n - 1)
    f = up - k
    last = k >= n - 1
    f[last] = 0.0
    rows = np.arange(len(u))
    W[rows, k] = 1.0 - f
    nxt = np.minimum(k + 1, n - 1)
    W[rows[~last], nxt[~last]] += f[~last]
    if ramp is None:
        ramp = min(1.0, n * 0.5)
    att = smoothstep(np.clip(u / max(float(ramp), 1e-9), 0.0, 1.0))
    return W, att


def carve(W, deform, M, shares, new_cols, parent_col):
    """CARVE blend (spec 5.7) on a dense weight block, in place.

    W: (T, G) weights of the touched vertices over all groups involved (existing + new + parent).
    deform: (G,) bool, existing deforming groups that may be carved (bone of the armature, use_deform, not locked,
    not one of new_cols). M: (T,) mask. shares: (T, nb) per new bone, rows sum to 1. new_cols: column of each new
    bone. parent_col: column of the parent bone group (gets 1 - M on vertices without any deforming weight), or an
    int array (T,) with one parent column per row."""
    M = np.asarray(M, dtype=np.float64)
    S = W[:, deform].sum(axis=1)
    others = W[:, ~deform].sum(axis=1)           # locked / non-bone weight, never changed
    W[:, deform] *= (1.0 - M)[:, None]
    has = S > 0.0
    room = np.clip(1.0 - others, 0.0, 1.0)       # S = 0: fill up to a total of 1, not on top of locked weight
    give = np.where(has, S * M, M * room)
    for j, col in enumerate(new_cols):
        W[:, col] += give * shares[:, j]
    if parent_col is not None:
        rows = np.flatnonzero(~has)
        pc = parent_col[rows] if np.ndim(parent_col) else parent_col
        W[rows, pc] += ((1.0 - M) * room)[rows]
    return W


def limit_influences(W, free, n_max):
    """Keep at most n_max non-zero weights per row among the 'free' columns; renormalise to the previous total of
    the free columns. free = groups of deform bones that are not locked. Every other column (locked groups,
    groups that are not deform bones: pin, mask, ...) is left alone and does not count."""
    if n_max <= 0:
        return W
    F = W[:, free]
    before = F.sum(axis=1)
    if F.shape[1] > n_max:
        order = np.argsort(-F, axis=1)
        drop = order[:, n_max:]
        rows = np.arange(len(F))[:, None]
        F[rows, drop] = 0.0
    after = F.sum(axis=1)
    ok = after > 0.0
    F[ok] *= (before[ok] / after[ok])[:, None]
    W[:, free] = F
    return W


def face_pairs(loop_verts, loop_poly, sel):
    """Pairs (v, w) of selected vertices that share a face (each selected corner with the first selected corner
    of its face): on n-gons two selected vertices can touch through a face without sharing an edge."""
    lv = np.asarray(loop_verts, dtype=np.int64)
    lp = np.asarray(loop_poly, dtype=np.int64)
    on = np.asarray(sel, dtype=bool)[lv]
    lv, lp = lv[on], lp[on]
    if not len(lv):
        return np.zeros((0, 2), dtype=np.int64)
    _u, first, inv = np.unique(lp, return_index=True, return_inverse=True)
    return np.stack([lv[first][inv], lv], axis=1)


def count_parts(n_verts, edges, sel, pairs=None):
    """Number of separate parts of the selection: connected components of the selected vertices over edges
    with both ends selected, plus the extra pairs (face_pairs). A selected vertex that touches no other
    selected vertex is a part of its own."""
    sel = np.asarray(sel, dtype=bool)
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    E = E[sel[E[:, 0]] & sel[E[:, 1]]]
    if pairs is not None and len(pairs):
        E = np.concatenate([E, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)])
    parent = np.arange(n_verts)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in E.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in np.flatnonzero(sel).tolist()})


def edge_step(edges, w):
    """Largest weight difference across an edge."""
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(E) == 0:
        return 0.0
    return float(np.max(np.abs(w[E[:, 0]] - w[E[:, 1]])))


def combine_sides(Ms, shares):
    """Merge the masks of several sides (mirror pair) into one CARVE: M = min(1, sum M_i), shares of every new
    bone = M_i * share_ij / sum M_i. Where the sides do not overlap (or sum M_i <= 1) each bone gets exactly
    S * M_i * share_ij, so a side's weight is as smooth as its own mask; carving the sides one after another
    multiplies two ramps where they overlap (mirror line) and steepens the weight there. Returns (M, shares,
    side) with side = index of the side with the largest mask per vertex (for the parent fill)."""
    Ms = np.stack([np.asarray(m, dtype=np.float64) for m in Ms], axis=1)
    tot = Ms.sum(axis=1)
    safe = np.where(tot > 0.0, tot, 1.0)
    S = np.concatenate([shares[i] * (Ms[:, i] / safe)[:, None] for i in range(len(shares))], axis=1)
    return np.minimum(1.0, tot), S, np.argmax(Ms, axis=1)


RADIUS_PERCENTILE = 85.0     # joint_radii: share of a bone's vertices inside its collision radius
RADIUS_MIN_WEIGHT = 0.5      # ... vertices with at least this much weight on the chain
RADIUS_MIN_VERTS = 8         # ... fewer vertices on a bone: not measured (0)


def joint_radii(P, W, J, thin=True):
    """Collision radius per bone of a chain from the mesh it carries (1.0.1). P (n,3) vertices, W (n,nb) their
    weights on the nb chain bones, J (nb+1,3) joints, all in one space. A vertex counts for the bone it has the
    most weight on, when its chain weights sum to RADIUS_MIN_WEIGHT or more. The radius is half the thickness of
    the mesh around the bone across its thin side: the offsets of the vertices from the bone are measured along
    the direction in which they spread least (a ponytail is wide and flat and lies on the back with its flat
    side; a round strand or tail has no thin side and gets its radius), and the radius is the offset that holds
    RADIUS_PERCENTILE of them (the bulk of a strand, not its stray hairs). Returns (nb,), 0 where fewer than
    RADIUS_MIN_VERTS vertices count. thin=False: the distance from the bone in any direction instead (how far
    the mesh reaches)."""
    nb = len(J) - 1
    out = np.zeros(nb)
    if not len(P) or nb < 1:
        return out
    use = W.sum(axis=1) >= RADIUS_MIN_WEIGHT
    own = np.argmax(W, axis=1)
    for j in range(nb):
        X = P[use & (own == j)]
        if len(X) < RADIUS_MIN_VERTS:
            continue
        a, ab = J[j], J[j + 1] - J[j]
        t = np.clip(along(X - a, ab) / max(float(ab @ ab), 1e-30), 0.0, 1.0)
        d = X - (a + t[:, None] * ab)
        if thin:
            _w, vec = np.linalg.eigh(d.T @ d)             # ascending: [0] is along the bone (offsets ~ 0 there)
            u = ab / max(float(np.linalg.norm(ab)), 1e-30)
            k = sorted(range(3), key=lambda i: abs(float(vec[:, i] @ u)))[:2]      # the two axes across the bone
            side = min(k)                                 # the smaller eigenvalue of the two
            out[j] = float(np.percentile(np.abs(along(d, vec[:, side])), RADIUS_PERCENTILE))
        else:
            out[j] = float(np.percentile(np.sqrt(np.einsum('nj,nj->n', d, d)), RADIUS_PERCENTILE))
    return out

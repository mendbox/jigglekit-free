# SPDX-License-Identifier: GPL-3.0-or-later
"""Jiggle physics on numpy (spec 5.2, 5.3). No bpy.

Every bone is a pendulum hinged at its head. Its tail (particle P) is pulled by a spring to the animated tail T,
damped relative to the reference velocity, given inertia and gravity, then constrained by angle, rigid length and
colliders. Position-based dynamics, semi-implicit Euler, root to tip; the tip never pulls the root.

Data layout. All N simulated bones live in flat arrays (chain after chain, root first). Parents of chain roots are
either simulated bones (another chain) or one of E external matrices (a non-simulated parent bone in world space,
or the armature matrix for a root without parent). One combined array of N + E rotations / translations holds the
world frames; a bone's parent is an index into it. Chains are grouped in waves (depth of chain nesting), and inside
a wave all chains with the same bone index are one numpy batch.

Rotations are 3x3 + vector, not 4x4 (spec 13, risk b). World frames carry the uniform world scale."""
import math

import numpy as np

try:
    from . import collide
    from .const import (CONSTRAINT_PASSES, DAMP_RATE, EPS_AXIS, K_BASE, K_CLAMP_OMEGA_DT, MAX_SUBSTEPS,
                        OMEGA_DT_MAX, ROT_INTERP_MAX)
except ImportError:          # imported as a top-level module (pure tests)
    import collide
    from const import (CONSTRAINT_PASSES, DAMP_RATE, EPS_AXIS, K_BASE, K_CLAMP_OMEGA_DT, MAX_SUBSTEPS,
                       OMEGA_DT_MAX, ROT_INTERP_MAX)

CHAIN_DEFAULTS = dict(stiffness=0.30, damping=0.14, inertia=0.90, gravity=0.0, angle_limit=math.radians(30.0),
                      tip_ratio=1.0, tip_damping=1.0, buckle=0.0, radius=0.0, collide=True, influence=1.0)
CHAIN_REF_BONES = 4.0        # chain factor: the sliders describe a chain of this many equal bones
CHAIN_COUNT_MIN = 2.0        # ... and a chain counts as at least this many bones
CHAIN_COUNT_MAX = 8.0        # ... and at most this many
SHORT_BONE_MAX = 4.0         # cap of the short-bone factor (longest bone of the chain / this bone)
BUCKLE_SEED = 0.05           # weight of the bone's own Z axis in the bow direction (a straight strand along gravity)


class Model:
    """Static data, built once per rebuild (spec 5.3)."""

    def __init__(self, chains, n_ext, substeps=2):
        self.n_ext = int(n_ext)
        self.substeps = int(substeps)
        nc = len(chains)
        lens = [len(c['lengths']) for c in chains]
        start = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        N = int(start[-1])
        self.N, self.n_chains = N, nc
        self.chain_start, self.chain_len = start[:-1], np.array(lens, dtype=np.int64)
        self.chain_of = np.repeat(np.arange(nc), lens).astype(np.int64)
        self.length = np.zeros(N)
        self.rrest_R = np.zeros((N, 3, 3))
        self.rrest_t = np.zeros((N, 3))
        self.par = np.zeros(N, dtype=np.int64)
        per = {k: np.zeros(N) for k in ('k', 'damping', 'inertia', 'gravity', 'limit', 'radius', 'influence',
                                        'buckle')}
        self.collide = np.zeros(N, dtype=bool)
        for ci, c in enumerate(chains):
            p = dict(CHAIN_DEFAULTS)
            p.update({k: v for k, v in c.items() if k in CHAIN_DEFAULTS})
            n = lens[ci]
            s0 = int(start[ci])
            total = float(sum(float(x) for x in c['lengths']))
            lmax = max((float(x) for x in c['lengths']), default=0.0)
            # Chain factor (chain-aware spring and damping). The spring, gravity and the parent's motion all act on
            # the tail as a distance, so a long bone bends a smaller angle than a short one for the same push, and
            # every bone adds its swing to the swing of the bones above it: with the same per-bone values a chain
            # of six short bones travels about twice as far as a chain of three long ones of the same length. So
            # the sliders describe a chain of CHAIN_REF_BONES equal bones (four: exactly the slider values), and
            # the spring and the damping rate of every bone are multiplied by (bone count) / CHAIN_REF_BONES.
            # Chains of two or three long bones get softer and ring longer, chains of many short bones get
            # stiffer and stop sooner; the tip travel per chain length comes out about the same (check
            # chain_count_travel; measured: at a fixed spring the damping rate that gives the same tip travel is
            # proportional to the bone count). The count is chain length / longest bone, so a stub at the end
            # does not count as a whole bone, kept between CHAIN_COUNT_MIN and CHAIN_COUNT_MAX (above eight
            # bones the factor made long tails too stiff). A single bone is not a chain: its values are the
            # slider values.
            cf = min(CHAIN_COUNT_MAX, max(CHAIN_COUNT_MIN, total / lmax)) / CHAIN_REF_BONES \
                if n > 1 and lmax > 1e-12 else 1.0
            last_head = total - float(c['lengths'][-1]) if n > 1 else 0.0
            head = 0.0
            for j in range(n):
                i = s0 + j
                rr = np.asarray(c['rrest'][j], dtype=np.float64)
                self.rrest_R[i] = rr[:3, :3]
                self.rrest_t[i] = rr[:3, 3]
                self.length[i] = float(c['lengths'][j])
                # position of the bone's head along the chain, 0 at the first bone, 1 at the last (by length;
                # equal bones: j / (n - 1))
                u = (head / last_head if last_head > 1e-12 else j / (n - 1)) if n > 1 else 0.0
                head += self.length[i]
                # Short bones. A bone shorter than the longest of its chain gets the spring of a pendulum of its
                # length (k x longest / length: the same sag angle as the longest bone) and the damping rate x
                # sqrt of that (the same damping ratio), or a short end bone whips and folds. Equal bones: 1.
                lr = min(SHORT_BONE_MAX, lmax / self.length[i]) if self.length[i] > 1e-12 else 1.0
                per['k'][i] = K_BASE * p['stiffness'] * (1.0 + (p['tip_ratio'] - 1.0) * u) * lr * cf
                # Tip Damping: the damping rate (-ln(1 - damping)) grows linearly along the chain from the first
                # bone to tip_damping times that at the last bone (as Tip Ratio does for the stiffness). The lower
                # bones of a soft chain are whipped by every bone above them; without this they fold into the
                # angle limit (docs/PRESETS.md, "Hair, tail and ears retuned").
                per['damping'][i] = 1.0 - (1.0 - p['damping']) ** ((1.0 + (p['tip_damping'] - 1.0) * u)
                                                                   * math.sqrt(lr) * cf)
                per['buckle'][i] = p['buckle']
                per['inertia'][i] = p['inertia']
                per['gravity'][i] = p['gravity']
                per['limit'][i] = p['angle_limit']
                per['radius'][i] = p['radius']
                per['influence'][i] = p['influence']
                self.collide[i] = bool(p['collide'])
                if j > 0:
                    self.par[i] = i - 1
            kind = c['parent'][0]
            if kind == 'ext':
                e = int(c['parent'][1])
                if not 0 <= e < self.n_ext:
                    raise ValueError("external parent index out of range")
                self.par[s0] = N + e
            elif kind == 'bone':
                pc, pb = int(c['parent'][1]), int(c['parent'][2])
                if not (0 <= pc < nc and 0 <= pb < lens[pc]):
                    raise ValueError("parent bone out of range")
                self.par[s0] = int(start[pc]) + pb
            else:
                raise ValueError("parent kind must be 'ext' or 'bone'")
        self.k = per['k']
        self.damping = per['damping']
        self.inertia = per['inertia']
        self.gravity = per['gravity']
        self.limit = per['limit']
        self.cos_limit = np.cos(per['limit'])
        self.sin_limit = np.sin(per['limit'])
        self.radius = per['radius']
        self.influence = per['influence']
        self.buckle = per['buckle']
        self.omega_max = float(np.sqrt(self.k.max())) if N else 0.0
        self.waves = self._waves(chains, lens)
        self.schedule = []
        for wave in self.waves:
            for j in range(max(lens[c] for c in wave)):
                idx = np.array([start[c] + j for c in wave if lens[c] > j], dtype=np.int64)
                self.schedule.append((idx, self.par[idx]))
        self._layout(lens, start)

    def _layout(self, lens, start):
        """Internal bone order for step(): wave by wave, bone index by bone index, and inside a wave the chains
        sorted by length (longest first). Every batch of the schedule is then one contiguous slice, and the
        parents of a batch with bone index j > 0 are the first rows of the batch j - 1: numpy views instead of
        gathers, in-place updates instead of temporary arrays (speed task 2026-11-23). State stays in the
        external (chain after chain) order; step() gathers it once per call and scatters it back."""
        N = self.N
        order = []
        batches = []
        for wave in self.waves:
            cs = sorted(wave, key=lambda c: -lens[c])            # stable: equal lengths keep their order
            prev = None
            for j in range(lens[cs[0]]):
                rows = [c for c in cs if lens[c] > j]
                s0 = len(order)
                order.extend(int(start[c]) + j for c in rows)
                sl = slice(s0, s0 + len(rows))
                batches.append((sl, j, prev, len(rows)))
                prev = sl
        perm = np.array(order, dtype=np.int64)              # internal position -> external bone index
        inv = np.empty(N, dtype=np.int64)
        inv[perm] = np.arange(N)
        self.perm, self.inv = perm, inv
        self.identity = bool(N == 0 or np.array_equal(perm, np.arange(N)))
        par_i = np.where(self.par[perm] >= N, self.par[perm], 0)
        sim_par = self.par[perm] < N
        par_i[sim_par] = inv[self.par[perm][sim_par]]       # parent in internal numbering (N + e for external)
        is_parent = np.zeros(N + self.n_ext, dtype=bool)
        is_parent[par_i] = True
        self.i_length = self.length[perm]
        self.i_rrest_R = np.ascontiguousarray(self.rrest_R[perm])
        self.i_rrest_t = self.rrest_t[perm]
        self.i_k = self.k[perm]
        self.i_damping = self.damping[perm]
        self.i_inertia = self.inertia[perm]
        self.i_keep = (1.0 - self.inertia[perm])[:, None]
        self.i_gravity = self.gravity[perm]
        self.i_cos = self.cos_limit[perm]
        self.i_sin = self.sin_limit[perm]
        self.i_radius = self.radius[perm]
        self.i_collide = self.collide[perm]
        self.i_buckle = self.buckle[perm]
        # step-level collider skip (sim._near_bones): roots of wave-0 chains, chain of every internal row
        self.i_chain = self.chain_of[perm]
        w0 = self.waves[0] if self.waves else []
        nested = [c for w in self.waves[1:] for c in w]
        self.w0_chains = np.array(w0, dtype=np.int64)
        self.w0_root = inv[self.chain_start[self.w0_chains]] if len(w0) else np.zeros(0, dtype=np.int64)
        self.w0_ext = self.par[self.chain_start[self.w0_chains]] - N if len(w0) else np.zeros(0, dtype=np.int64)
        self.w0_root_ext = self.chain_start[self.w0_chains]
        self.nested = [(c, int(self.chain_of[self.par[self.chain_start[c]]])) for c in nested]
        self.chain_rmax = np.array([self.radius[s0:s0 + n].max() if n else 0.0
                                    for s0, n in zip(self.chain_start, self.chain_len)])
        self.batches = []
        for sl, j, prev, n in batches:
            if j > 0:
                par = slice(prev.start, prev.start + n)             # first n rows of the previous batch
            else:
                p = par_i[sl]
                par = slice(int(p[0]), int(p[0]) + n) if np.array_equal(p, np.arange(p[0], p[0] + n)) else p
            act = self.i_collide[sl]
            bk = self.i_buckle[sl]
            self.batches.append(dict(sl=sl, par=par, n=n, children=bool(is_parent[sl].any()),
                                     buckle=bk[:, None] if bk.any() else None,
                                     collide=bool(act.any()), act=None if act.all() else act,
                                     cos=self.i_cos[sl], sin=self.i_sin[sl]))

    @staticmethod
    def _waves(chains, lens):
        nc = len(chains)
        depth = [-1] * nc

        def d(ci, seen):
            if depth[ci] >= 0:
                return depth[ci]
            if ci in seen:
                raise ValueError("chain parent loop")
            par = chains[ci]['parent']
            depth[ci] = 0 if par[0] == 'ext' else 1 + d(int(par[1]), seen + (ci,))
            return depth[ci]

        for ci in range(nc):
            d(ci, ())
        out = []
        for w in range(max(depth) + 1 if nc else 0):
            out.append([ci for ci in range(nc) if depth[ci] == w and lens[ci] > 0])
        return [w for w in out if w]


class Frame:
    """Inputs of one frame (read from the evaluated armature by runtime.py, or made by tests).
    ext (E,4,4): world matrices of external parents; banim (N,4,4): matrix_basis of simulated bones;
    q_anim (N,4): their authored quaternions (w, x, y, z); colliders: col_a, col_b (C,3) world points and
    col_r (C,) world radii; scale: uniform world scale; gravity (3,) in BU/s^2."""

    def __init__(self, ext, banim, q_anim, col_a=None, col_b=None, col_r=None, scale=1.0,
                 gravity=(0.0, 0.0, 0.0), fps=24.0, time_scale=1.0):
        self.ext = np.asarray(ext, dtype=np.float64).reshape(-1, 4, 4)
        self.banim = np.asarray(banim, dtype=np.float64).reshape(-1, 4, 4)
        self.q_anim = np.asarray(q_anim, dtype=np.float64).reshape(-1, 4)
        z = np.zeros((0, 3))
        self.col_a = z if col_a is None else np.asarray(col_a, dtype=np.float64).reshape(-1, 3)
        self.col_b = self.col_a if col_b is None else np.asarray(col_b, dtype=np.float64).reshape(-1, 3)
        self.col_r = np.zeros(len(self.col_a)) if col_r is None else np.asarray(col_r, dtype=np.float64).ravel()
        self.scale = float(scale)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.fps = float(fps)
        self.time_scale = float(time_scale)


class State:
    def __init__(self, N, E):
        self.P = np.zeros((N, 3))
        self.V = np.zeros((N, 3))
        self.Tprev = np.zeros((N, 3))
        self.T = np.zeros((N, 3))       # animated tail of the last substep (NaN reset)
        self.H = np.zeros((N, 3))       # head of the last substep
        self.q_prev = np.zeros((N, 4))
        self.q_prev[:, 0] = 1.0
        self.ext_R = np.zeros((E, 3, 3))
        self.ext_t = np.zeros((E, 3))
        self.col_a = np.zeros((0, 3))
        self.col_b = np.zeros((0, 3))
        self.valid = False
        self.nan_resets = 0
        self.contacts = 0
        self.n_sub = 0


# --------------------------------------------------------------------------------------------- small math helpers

def _norm(v):
    return np.sqrt(np.einsum('...j,...j->...', v, v))


def _normalize(v, fallback=None):
    n = _norm(v)
    ok = n > 1e-300
    out = np.zeros_like(v)
    out[ok] = v[ok] / n[ok][:, None]
    if fallback is not None and not ok.all():
        out[~ok] = fallback[~ok]
    return out


def _mv(R, v):
    """Batched R (n,3,3) @ v (n,3)."""
    return np.einsum('nij,nj->ni', R, v)


def _arc(da, ds):
    """Shortest-arc rotation from unit da to unit ds (Rodrigues). Returns R (n,3,3), unit axis u (n,3), angle (n,).
    Axis length under EPS_AXIS: identity."""
    a = np.cross(da, ds)
    s = _norm(a)
    c = np.einsum('nj,nj->n', da, ds)
    ok = s > EPS_AXIS
    th = np.where(ok, np.arctan2(s, c), 0.0)
    u = np.zeros_like(a)
    u[ok] = a[ok] / s[ok][:, None]
    ct, st = np.cos(th), np.sin(th)
    n = len(da)
    R = np.empty((n, 3, 3))
    R[:] = np.eye(3)
    R *= ct[:, None, None]
    R += (1.0 - ct)[:, None, None] * (u[:, :, None] * u[:, None, :])
    x, y, z = u[:, 0] * st, u[:, 1] * st, u[:, 2] * st
    R[:, 0, 1] -= z
    R[:, 0, 2] += y
    R[:, 1, 0] += z
    R[:, 1, 2] -= x
    R[:, 2, 0] -= y
    R[:, 2, 1] += x
    return R, u, th


def qmul(a, b):
    """Hamilton product, (w, x, y, z), batched."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def _perp_any(d):
    """A unit vector perpendicular to each unit d: cross with the world axis least aligned with d."""
    ax = np.argmin(np.abs(d), axis=1)
    e = np.zeros_like(d)
    e[np.arange(len(d)), ax] = 1.0
    return _normalize(np.cross(d, e))


def _col_lengths(R):
    return np.sqrt(np.einsum('nij,nij->nj', R, R))     # column lengths (n,3)


# ----------------------------------------------------------------------------------------------------- the solver

def _lloc(m, fr):
    """Lloc = Rrest @ Banim as rotation (N,3,3) and translation (N,3)."""
    BR = fr.banim[:, :3, :3]
    Bt = fr.banim[:, :3, 3]
    Rl = np.matmul(m.rrest_R, BR)
    tl = _mv(m.rrest_R, Bt) + m.rrest_t
    return Rl, tl


def _ext_interp_setup(st, fr):
    """Current external frames and which of them rotate more than ROT_INTERP_MAX this frame."""
    Rc = fr.ext[:, :3, :3]
    tc = fr.ext[:, :3, 3]
    if not st.valid or len(Rc) == 0:
        return Rc, tc, np.ones(len(Rc), dtype=bool), _col_lengths(Rc) if len(Rc) else np.zeros((0, 3))
    lc = _col_lengths(Rc)
    lp = _col_lengths(st.ext_R)
    Rcn = Rc / np.maximum(lc[:, None, :], 1e-300)
    Rpn = st.ext_R / np.maximum(lp[:, None, :], 1e-300)
    tr = np.einsum('nij,nij->n', Rpn, Rcn)                       # trace(Rp^T Rc)
    big = tr < 1.0 + 2.0 * math.cos(ROT_INTERP_MAX)
    return Rc, tc, big, lc


def reset(m, st, fr):
    """Initial state on the current animated pose (spec 5.2): P = T, V = 0. Returns q_anim (N,4)."""
    N, E = m.N, m.n_ext
    Rl, tl = _lloc(m, fr)
    R_all = np.empty((N + E, 3, 3))
    t_all = np.empty((N + E, 3))
    R_all[N:] = fr.ext[:, :3, :3]
    t_all[N:] = fr.ext[:, :3, 3]
    for idx, par in m.schedule:
        Rp, tp = R_all[par], t_all[par]
        Ra = np.matmul(Rp, Rl[idx])
        H = _mv(Rp, tl[idx]) + tp
        T = H + Ra[:, :, 1] * m.length[idx][:, None]
        R_all[idx] = Ra
        t_all[idx] = H
        st.H[idx] = H
        st.T[idx] = T
    st.P[:] = st.T
    st.Tprev[:] = st.T
    st.V[:] = 0.0
    st.ext_R = fr.ext[:, :3, :3].copy()
    st.ext_t = fr.ext[:, :3, 3].copy()
    st.col_a = fr.col_a.copy()
    st.col_b = fr.col_b.copy()
    st.q_prev = fr.q_anim.copy()
    st.valid = True
    st.n_sub = 0
    return fr.q_anim.copy()


def substeps_for(m, dt_f):
    if m.N == 0:
        return 1
    return int(min(max(m.substeps, math.ceil(m.omega_max * dt_f / OMEGA_DT_MAX), 1), MAX_SUBSTEPS))


def step(m, st, fr, n_frames=1):
    """Advance n_frames frames (one step of dt_f = time_scale * n_frames / fps). Returns q (N,4) to write as
    rotation_quaternion of the simulated bones. On the first call (state not valid) this is reset().

    Works in the internal bone order of Model._layout (contiguous batches, views, in-place updates); the maths
    per bone is spec 5.2 as before."""
    if not st.valid:
        return reset(m, st, fr)
    N, E = m.N, m.n_ext
    if N == 0:
        return np.zeros((0, 4))
    dt_f = fr.time_scale * n_frames / fr.fps
    n_sub = substeps_for(m, dt_f)
    dt = dt_f / n_sub
    inv_dt = 1.0 / dt
    st.n_sub = n_sub
    perm = m.perm
    # per-call constants, internal order, columns ready for broadcasting
    k = np.minimum(m.i_k, (K_CLAMP_OMEGA_DT / dt) ** 2)
    ret = (1.0 - m.i_damping) ** (DAMP_RATE * dt)
    c_ret = ret[:, None]
    c_ref = ((1.0 - ret) * m.i_inertia * inv_dt)[:, None]     # V_ref (1 - ret) = inertia delta / dt (1 - ret)
    c_k = (dt * k)[:, None]
    dt_ag = (dt * m.i_gravity)[:, None] * fr.gravity[None, :]
    keep = m.i_keep
    gn = float(np.sqrt(fr.gravity @ fr.gravity))
    g_hat = fr.gravity / gn if gn > 1e-12 else np.zeros(3)
    # Lloc = Rrest @ Banim as one 3x5 block per bone: [R | t | t + length * R_y] (rotation, head, animated tail)
    banim = fr.banim if m.identity else fr.banim[perm]
    LL = np.empty((N, 3, 5))
    np.matmul(m.i_rrest_R, banim[:, :3, :], out=LL[:, :, :4])
    LL[:, :, 3] += m.i_rrest_t
    np.multiply(LL[:, :, 1], m.i_length[:, None], out=LL[:, :, 4])
    LL[:, :, 4] += LL[:, :, 3]
    Rc, tc, big, lc = _ext_interp_setup(st, fr)
    Rp0, tp0 = st.ext_R, st.ext_t
    C = len(fr.col_a)
    use_col = C > 0 and bool(m.collide.any())
    if use_col:
        if len(st.col_a) != C:
            st.col_a, st.col_b = fr.col_a.copy(), fr.col_b.copy()
        col_r = fr.col_r
        radius = m.i_radius * fr.scale
        near_rows = _near_bones(m, st, fr, LL, Rc, tc, big, lc)
        if near_rows is None:
            use_col = False                               # no collider can reach any chain in this step
    if m.identity:
        P, V, Tprev = st.P, st.V, st.Tprev
    else:
        P, V, Tprev = st.P[perm], st.V[perm], st.Tprev[perm]
    R_all = np.empty((N + E, 3, 3))
    t_all = np.empty((N + E, 3))
    Rp_last = np.empty((N, 3, 3))
    H_last = np.empty((N, 3))
    T_last = np.empty((N, 3))
    da_last = np.empty((N, 3))
    ds_last = np.empty((N, 3))
    contacts = 0
    for s in range(n_sub):
        alpha = (s + 1) / n_sub
        last = s == n_sub - 1
        if last or E == 0:
            R_all[N:] = Rc
            t_all[N:] = tc
        else:
            Ri = Rp0 + alpha * (Rc - Rp0)
            Ri *= (lc / np.maximum(_col_lengths(Ri), 1e-300))[:, None, :]
            Ri[big] = Rc[big]
            R_all[N:] = Ri
            t_all[N:] = tp0 + alpha * (tc - tp0)
        if use_col:
            if last:
                A, B = fr.col_a, fr.col_b
            else:
                A = st.col_a + alpha * (fr.col_a - st.col_a)
                B = st.col_b + alpha * (fr.col_b - st.col_b)
            AB, inv_ab2 = collide.segments(A, B)
        for bt in m.batches:
            sl = bt['sl']
            par = bt['par']
            Rp, tp = R_all[par], t_all[par]
            M = np.matmul(Rp, LL[sl])
            Ra = M[:, :, :3]
            H = M[:, :, 3] + tp
            T = M[:, :, 4] + tp
            D = T - H
            L = np.sqrt(np.einsum('nj,nj->n', D, D))
            da = _unit_rows(D, L, None)
            # 2. inertia shift
            Pi = P[sl]                                   # view: updated in place
            Tp = Tprev[sl]
            spring = Tp - Pi                             # offset at the start of the substep (see below)
            delta = T - Tp
            Tp[:] = T
            Pi += keep[sl] * delta
            Pold = Pi.copy()
            # 3. velocity, damped relative to Vref = inertia delta / dt; 4. position
            Vi = V[sl]
            Vi *= c_ret[sl]
            Vi += c_ref[sl] * delta
            Vi += dt_ag[sl]
            push = None
            if bt['buckle'] is not None:
                push = _buckle(Pold, Vi, dt, H, L, Ra, g_hat, bt['buckle'])
            # spring on the offset before the target moved (T_prev - P_prev == T - P - inertia delta): symplectic
            # Euler in the target's frame. The spec text uses T - P after the inertia shift, which leaves a steady
            # lead of inertia * delta per substep at constant velocity (56 mm at 3 m/s, 24 fps, 2 substeps) and
            # makes the result depend on fps and substeps (checks const_velocity, fps_equiv, substeps_equiv).
            # Static target: delta = 0, identical to the spec formula (period, damping, presets unchanged).
            spring *= c_k[sl]
            Vi += spring
            Pi += dt * Vi
            if push is not None:
                Pi += push
            # 5. constraints: angle limit, rigid length, colliders (+ length again), two passes
            check = use_col and bt['collide'] and near_rows[sl].any()
            for _ in range(CONSTRAINT_PASSES):
                _limit_and_length(Pi, H, da, L, bt['cos'], bt['sin'])
                if not check:
                    break                    # nothing moved P after the projection: a second pass changes nothing
                Pr, nc = collide.resolve(H, Pi, radius[sl], A, AB, inv_ab2, col_r, bt['act'])
                if not nc:
                    break
                contacts += nc
                v = Pr - H
                Pi[:] = H + _unit_rows(v, np.sqrt(np.einsum('nj,nj->n', v, v)), da) * L[:, None]
            # 6. velocity from position
            np.subtract(Pi, Pold, out=Vi)
            Vi *= inv_dt
            # 7. simulated frame
            ds = Pi - H
            ds /= L[:, None]                             # |P - H| = L after the last projection
            if bt['children']:
                np.matmul(_arc_matrix(da, ds), Ra, out=R_all[sl])
                t_all[sl] = H
            if last:
                Rp_last[sl] = Rp
                H_last[sl] = H
                T_last[sl] = T
                da_last[sl] = da
                ds_last[sl] = ds
    if not m.identity:
        inv = m.inv
        st.P[perm] = P
        st.V[perm] = V
        st.Tprev[perm] = Tprev
        Rp_last, H_last, T_last, da_last, ds_last = Rp_last[inv], H_last[inv], T_last[inv], da_last[inv], ds_last[inv]
    st.H[:] = H_last
    st.T[:] = T_last
    st.ext_R = Rc.copy()
    st.ext_t = tc.copy()
    if use_col:
        st.col_a, st.col_b = fr.col_a.copy(), fr.col_b.copy()
    st.contacts = contacts
    u, th = _axis_angle(da_last, ds_last)
    q = _outputs(m, st, fr, Rp_last, u, th)
    _nan_guard(m, st, fr, q)
    st.q_prev = q.copy()
    return q


def _buckle(Pold, V, dt, H, L, Ra, g_hat, gain):
    """Bending under a load along the bone (Buckle). A rigid pendulum ignores every force along its own axis: an
    upright strand does not react to a hop, a landing or its own weight, because the length constraint removes
    that part of the motion. Here the part it would remove is measured before the spring acts: how far gravity,
    the parent's acceleration and the carried velocity push the tail towards the head in this substep (c >= 0;
    a pull is left alone, a taut strand stays straight). A share of it is given back sideways, down the slope of
    gravity: gain * c * (how far the bone points up, 0 for a level or hanging bone, 1 for an upright one). An
    upright bone has no slope: it bows towards its own Z axis (BUCKLE_SEED). The share fades to nothing at
    level, so nothing switches when a bone swings through the horizontal; hanging bones are pendulums as before.
    Without a gravity direction there is no bow.
    c is acceleration * dt^2, so the sideways push is an acceleration of gain * (the load along the bone):
    independent of substeps and frame rate like gravity. Returns the displacement (n,3) or None."""
    w = Pold + dt * V
    w -= H
    r = np.sqrt(np.einsum('nj,nj->n', w, w))
    c = L - r
    w /= np.maximum(r, 1e-300)[:, None]
    up = -(w @ g_hat)                                        # 1 = the bone points straight up, < 0 = it hangs
    c *= up
    if c.max() <= 0.0 or up.max() <= 0.0:
        return None
    c[(c < 0.0) | (up < 0.0)] = 0.0
    z = Ra[:, :, 2]
    b = z * (BUCKLE_SEED / np.maximum(np.sqrt(np.einsum('nj,nj->n', z, z)), 1e-300))[:, None]
    b += g_hat
    b -= w * np.einsum('nj,nj->n', b, w)[:, None]
    b *= (gain[:, 0] * c / np.sqrt(np.einsum('nj,nj->n', b, b) + BUCKLE_SEED ** 2))[:, None]
    return b


def _unit_rows(v, n, fallback):
    """v / n row by row; rows with n <= 1e-300 get the fallback row (or 0)."""
    if n.min() > 1e-300:
        return v / n[:, None]
    ok = n > 1e-300
    out = np.zeros_like(v) if fallback is None else fallback.copy()
    out[ok] = v[ok] / n[ok][:, None]
    return out


def _limit_and_length(P, H, da, L, cos_lim, sin_lim):
    """Angle limit, then rigid length, in place on P (rows of one batch). Same result as _angle_limit followed
    by the length projection, in fewer numpy calls: rows over the limit get the direction on the limit cone in
    the plane of da and their direction, all rows get length L."""
    v = P - H
    nv = np.sqrt(np.einsum('nj,nj->n', v, v))
    if nv.min() <= 1e-300:                                   # degenerate rows: the general path
        P[:] = _angle_limit(P, H, da, L, cos_lim, sin_lim)
        P[:] = H + _normalize(P - H, da) * L[:, None]
        return
    ds = v / nv[:, None]
    c = np.einsum('nj,nj->n', da, ds)
    over = c < cos_lim
    if over.any():
        perp = ds - da * c[:, None]
        pn = np.sqrt(np.einsum('nj,nj->n', perp, perp))
        if pn[over].min() < 1e-9:                            # near 180 degrees: the general path
            P[:] = _angle_limit(P, H, da, L, cos_lim, sin_lim)
            P[:] = H + _normalize(P - H, da) * L[:, None]
            return
        perp *= (sin_lim / np.maximum(pn, 1e-300))[:, None]
        perp += da * cos_lim[:, None]
        ds = np.where(over[:, None], perp, ds)
    np.multiply(ds, L[:, None], out=P)
    P += H


def _near_bones(m, st, fr, LL, Rc, tc, big, lc):
    """Step-level collider skip: rows (internal order) of the chains that a collider can reach during this step,
    or None when no collider can reach any chain. Conservative bounding spheres: a chain stays within its world
    length (+ 1 %, + bone radius) of its root head, the root head moves between its positions at the start and the
    end of the step (+ the room the column-normalised rotation interpolation leaves); a collider stays inside the
    sphere around its four end points (start and end of the step) + its radius. Nested chains use the sphere of
    their parent chain grown by their own length. Skipping only removes tests that cannot find a contact."""
    if not len(m.w0_chains):
        return None
    # world length bound per chain: ext scale x the largest scale along the chain x sum of local lengths
    seg = LL[:, :, 4] - LL[:, :, 3]
    lw = np.sqrt(np.einsum('nj,nj->n', seg, seg))
    smax = max(float(_col_lengths(LL[:, :, :3]).max()), 1.0) if m.N else 1.0
    lw_ext = lw if m.identity else lw[m.inv]
    S = np.add.reduceat(lw_ext, m.chain_start) * (smax ** m.chain_len.astype(np.float64))
    e = m.w0_ext
    s_e = lc[e].max(axis=1) if len(lc) else np.ones(len(e))
    tl = LL[m.w0_root, :, 3]
    H1 = np.einsum('nij,nj->ni', Rc[e], tl) + tc[e]
    H0 = np.einsum('nij,nj->ni', st.ext_R[e], tl) + st.ext_t[e]
    dev = np.where(big[e], 2.0, 0.15) * s_e * np.sqrt(np.einsum('nj,nj->n', tl, tl))
    ctr = np.zeros((m.n_chains, 3))
    rad = np.zeros(m.n_chains)
    c0 = m.w0_chains
    ctr[c0] = 0.5 * (H0 + H1)
    d01 = H1 - H0
    rad[c0] = 0.5 * np.sqrt(np.einsum('nj,nj->n', d01, d01)) + dev + 1.01 * s_e * S[c0] \
        + m.chain_rmax[c0] * fr.scale
    for c, pc in m.nested:                                 # parents come first (waves are in order)
        ctr[c] = ctr[pc]
        rad[c] = rad[pc] + 1.01 * float(s_e.max()) * S[c] + m.chain_rmax[c] * fr.scale
    pts = np.stack((st.col_a, st.col_b, fr.col_a, fr.col_b))           # (4, C, 3)
    cc = pts.mean(axis=0)
    dd = pts - cc[None]
    cr = np.sqrt(np.einsum('kcj,kcj->kc', dd, dd)).max(axis=0) + fr.col_r
    dc = ctr[:, None, :] - cc[None, :, :]
    near = (np.einsum('ncj,ncj->nc', dc, dc) < np.square(rad[:, None] + cr[None, :])).any(axis=1)
    near &= np.add.reduceat(m.collide.astype(np.int64), m.chain_start) > 0 if m.n_chains else near
    if not near.any():
        return None
    return near[m.i_chain]


_I3 = np.eye(3)
_C1 = np.array([1, 2, 0])
_C2 = np.array([2, 0, 1])


def _arc_matrix(da, ds):
    """Shortest-arc rotation from unit da to unit ds: R = I + K + K K / (1 + c) with K = ds da^T - da ds^T
    (the cross-product matrix of da x ds) and c = da . ds. Same rotation as Rodrigues in _arc, without
    trigonometry. Near 180 degrees: _arc."""
    c = np.einsum('nj,nj->n', da, ds)
    if c.min() < -0.999:
        return _arc(da, ds)[0]
    K = ds[:, :, None] * da[:, None, :]
    K -= da[:, :, None] * ds[:, None, :]
    R = np.matmul(K, K)
    R *= (1.0 / (1.0 + c))[:, None, None]
    R += K
    R += _I3
    return R


def _axis_angle(da, ds):
    """Unit axis and angle of the shortest arc from da to ds (as _arc, without the matrix)."""
    a = da[:, _C1] * ds[:, _C2] - da[:, _C2] * ds[:, _C1]
    s = _norm(a)
    c = np.einsum('nj,nj->n', da, ds)
    ok = s > EPS_AXIS
    th = np.where(ok, np.arctan2(s, c), 0.0)
    u = np.zeros_like(a)
    u[ok] = a[ok] / s[ok][:, None]
    return u, th


def _angle_limit(P, H, da, L, cos_lim, sin_lim):
    v = P - H
    ds = _normalize(v, da)
    c = np.einsum('nj,nj->n', da, ds)
    over = c < cos_lim
    if not over.any():
        return P
    P = P.copy()
    o = np.nonzero(over)[0]
    dao, dso, co = da[o], ds[o], c[o]
    perp = dso - dao * co[:, None]
    pn = _norm(perp)
    deg = pn < 1e-9
    w = np.zeros_like(perp)
    ok = ~deg
    w[ok] = perp[ok] / pn[ok][:, None]
    if deg.any():                                  # near 180 degrees (near 0 cannot be over the limit)
        w[deg] = _perp_any(dao[deg])
    dnew = dao * cos_lim[o][:, None] + w * sin_lim[o][:, None]
    P[o] = H[o] + dnew * L[o][:, None]
    return P


def _outputs(m, st, fr, Rp_last, u, th):
    """Local rotation of every bone (spec 5.2, output): q_final = q_corr (x) q_anim in the parent frame at rest."""
    F = np.matmul(Rp_last, m.rrest_R)
    a_loc = _normalize(np.einsum('nji,nj->ni', F, u))          # F^T u: F is rotation x uniform scale
    # mirrored frame (negative scale on the object or a parent, det F < 0): a rotation about u seen from a
    # left-handed frame turns the other way, M^-1 R(u, th) M = R(det(M) M^-1 u, th), so the local axis flips
    mirrored = np.einsum('nj,nj->n', np.cross(F[:, :, 0], F[:, :, 1]), F[:, :, 2]) < 0.0
    if mirrored.any():
        a_loc[mirrored] *= -1.0
    h = 0.5 * th
    qc = np.empty((m.N, 4))
    qc[:, 0] = np.cos(h)
    qc[:, 1:] = a_loc * np.sin(h)[:, None]
    qa = fr.q_anim
    q = qmul(qc, qa)
    q /= _norm(q)[:, None]
    flip = np.einsum('nj,nj->n', q, st.q_prev) < 0.0
    q[flip] *= -1.0
    inf = m.influence
    if (inf < 1.0).any():
        qa2 = qa.copy()
        qa2[np.einsum('nj,nj->n', qa2, q) < 0.0] *= -1.0
        mix = qa2 + (q - qa2) * inf[:, None]
        q = np.where((inf < 1.0)[:, None], mix / _norm(mix)[:, None], q)
    return q


def _nan_guard(m, st, fr, q):
    bad = ~(np.isfinite(st.P).all(1) & np.isfinite(st.V).all(1) & np.isfinite(q).all(1))
    if not bad.any():
        return
    chains = np.unique(m.chain_of[bad])
    for ci in chains:
        s0 = int(m.chain_start[ci])
        sl = slice(s0, s0 + int(m.chain_len[ci]))
        T = st.T[sl]
        if not np.isfinite(T).all():
            T = np.nan_to_num(T)
        st.P[sl] = T
        st.Tprev[sl] = T
        st.V[sl] = 0.0
        q[sl] = fr.q_anim[sl]
    st.nan_resets += len(chains)


# ------------------------------------------------------------------------------- independent helpers for testing

def quat_to_mat(q):
    """(n,4) unit quaternions -> (n,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime of the live simulation (spec 5.3, 5.5): reading inputs from the evaluated armature, one physics step,
writing rotation_quaternion back to the original armature. Chains come from stored data (model.rebuild) or are
attached by code (attach(), tests).

Reads: foreach_get on ob_eval.pose.bones (matrix_basis, rotation_quaternion), direct access for the few external
parents and collider bones. Writes: foreach_get + foreach_set of rotation_quaternion on the original pose, then
ob.update_tag(), see docs/PROTOTYPE-NOTES.md (section 12 items 1 and 2).

Cache (spec 5.5): every computed frame is stored as (input signature, q, P, V, T) in float32, at most
cache_max_frames per armature (oldest dropped). A jump (backwards, or forwards by more than max_catchup) to a
cached frame with the same signature restores the pose and the swing state, so scrubbing back is instant and
playing on from there continues the swing; a jump to an uncached frame restarts on the animated pose. The cache is
dropped with the runtime on every parameter change (version), cleared when an Action changes (ANIM, bumped by
depsgraph_update_post) and not used while baking (recording)."""
import statistics
import time
from collections import deque

import bpy
import numpy as np

from . import sim

RT = {}            # key(object) -> ArmRT (the pointer is checked too: rt.ptr)
STATS = dict(pre_calls=0, post_calls=0)
BAKING = set()     # armature names being baked: simulated even with Live off (bake.record)
ANIM = [0]         # anim_version: bumped when an Action changed (handlers.on_depsgraph); clears every cache
MEDIAN_FRAMES = 30         # status line: rolling median of the simulation time over this many frames
SLOW_MS = 20.0             # above this median the panel suggests Bake (Pavel's request 2026-11-23)
_REDRAW = [0.0]
LOW_FPS_NOTE = "Frame rate is very low for stable simulation (under 6 fps or Time Scale over 3)."
REST_SPHERE = 1.0          # radius of a rest sphere / length of its chain (set_rest_spheres)
COLLIDER_OFF = -1.0e9      # radius of a collider whose object is gone: no contact is possible (not a point at 0)
HIST_FRAMES = 4            # results of the last integer frames kept for sub-frame evaluations (motion blur)
RESCAN_CALLS = 48          # the armature list of a scene is rebuilt at least every this many handler calls
_ARMS = {}                 # scene name -> [object count, calls since the scan, [armature objects]]
SCANS = [0]                # number of full scene scans (check active_cache)


def key(ob):
    """Runtime key of an armature object: its name; for an object linked from a library the library path is
    added, because a linked and a local object (or objects of two libraries) can have the same name."""
    lib = ob.library
    return ob.name if lib is None else "%s\0%s" % (ob.name, lib.filepath)


def invalidate():
    """The cached armature lists are stale (chains added or removed, undo, load, a parameter edit)."""
    _ARMS.clear()


def _m4(m):
    return np.array(m, dtype=np.float64)        # mathutils.Matrix -> rows


class ArmRT:
    def __init__(self, ob, chains, colliders=(), substeps=2, gravity_vector=(0.0, 0.0, -9.81), time_scale=1.0,
                 max_catchup=3, tag='DATA', rest_q=None):
        self.name = ob.name
        self.key = key(ob)
        self.enabled = True
        self.busy = False
        self.max_catchup = max_catchup
        self.time_scale = time_scale
        self.gravity_vector = np.array(gravity_vector, dtype=np.float64)
        self.tag = tag
        pbs = ob.pose.bones
        index = {pb.name: i for i, pb in enumerate(pbs)}
        self.n_pose = len(pbs)
        sim_names = [n for c in chains for n in c['bones']]
        where = {}
        for ci, c in enumerate(chains):
            for bi, n in enumerate(c['bones']):
                where[n] = (ci, bi)
        ext_keys = []
        mchains = []
        for ci, c in enumerate(chains):
            lengths, rrest = [], []
            for bi, n in enumerate(c['bones']):
                pb = pbs[n]
                b = pb.bone
                if b.parent is not None:
                    rr = b.parent.matrix_local.inverted() @ b.matrix_local
                else:
                    rr = b.matrix_local.copy()
                lengths.append(b.length)
                rrest.append(_m4(rr))
                if bi > 0 and (b.parent is None or b.parent.name != c['bones'][bi - 1]):
                    raise ValueError("chain bones must be parent -> child: %s" % n)
            root = pbs[c['bones'][0]].bone
            if root.parent is None:
                ekey = -1
            elif root.parent.name in where:
                ekey = None
            else:
                ekey = index[root.parent.name]
            if ekey is None:
                pc, pb_i = where[root.parent.name]
                parent = ('bone', pc, pb_i)
            else:
                if ekey not in ext_keys:
                    ext_keys.append(ekey)
                parent = ('ext', ext_keys.index(ekey))
            mc = {k: v for k, v in c.items() if k != 'bones'}
            mc.update(lengths=lengths, rrest=rrest, parent=parent)
            mchains.append(mc)
        self.ext_keys = ext_keys
        self.model = sim.Model(mchains, len(ext_keys), substeps=substeps)
        self.state = sim.State(self.model.N, self.model.n_ext)
        self.sim_idx = np.array([index[n] for n in sim_names], dtype=np.int64)
        q = np.zeros(self.n_pose * 4, dtype=np.float32)
        pbs.foreach_get("rotation_quaternion", q)
        self.rest_q = q.reshape(-1, 4)[self.sim_idx].astype(np.float32)
        if rest_q is not None:                    # stored PoseBone.jigglekit.rest_q (never the last written pose)
            self.rest_q = np.asarray(rest_q, dtype=np.float32).reshape(-1, 4)
        self.colliders = []
        for c in colliders:
            if c.get('object') or c.get('object_index') is not None:
                # stored colliders: index into Object.jigglekit.colliders, the object is read from the pointer
                # property on every frame (a rename, or two objects with one name, cannot break it);
                # colliders attached by code (tests): the object name
                self.colliders.append(dict(obj=c.get('object'), ci=c.get('object_index'), i=-1, len=0.0,
                                           r=float(c['radius']), t0=0.0, t1=0.0,
                                           off=np.array(c.get('offset', (0.0, 0.0, 0.0)), dtype=np.float64),
                                           shape='SPHERE'))
                continue
            pb = pbs[c['bone']]
            self.colliders.append(dict(i=index[c['bone']], len=pb.bone.length, r=float(c['radius']),
                                       t0=float(c.get('t0', 0.0)),
                                       t1=float(c.get('t1', 1.0 if c.get('shape', 'CAPSULE') == 'CAPSULE' else 0.0)),
                                       off=np.array(c.get('offset', (0.0, 0.0, 0.0)), dtype=np.float64),
                                       shape=c.get('shape', 'CAPSULE')))
        self.sim_names = sim_names
        self.set_rest(ob)
        self.set_rest_spheres(ob, chains, index)
        self.buf16 = np.zeros(self.n_pose * 16, dtype=np.float32)
        self.buf4 = np.zeros(self.n_pose * 4, dtype=np.float32)
        self.last_frame = None
        self.last_q = None
        self.last_written = None       # what write() put into the pose last (float32), for Capture Rest Pose
        self.hist = {}                 # integer frame -> q of the last HIST_FRAMES computed frames (sub-frames)
        self.resets = 0                # restarts on the animated pose (checks subframes, view_layers)
        self.contacts_total = 0
        self.timings = deque(maxlen=2000)     # per post call: (read, sim, write) seconds
        self.pre_times = deque(maxlen=2000)
        self.sim_ms = deque(maxlen=MEDIAN_FRAMES)
        self.sim_names = sim_names
        self.notes = []
        self.error = ""                # first error message of the frame handler (spec 5.5 step 0)
        self.errors = 0                # number of frames that raised
        self.root_checked = False      # 'chain root inside a collider' note, checked once per runtime
        self.from_props = False
        self.ptr = ob.as_pointer()
        self.built_version = None
        self.use_cache = True
        self.cache_max = 5000
        self.cache = {}                # frame -> (signature, q, P, V, T), float32
        self.cache_anim = ANIM[0]
        self.cache_hits = 0
        self.last_sig = None
        self.recording = False         # bake: no cache, no frame_start restart, q of every frame into record
        self.record = {}

    def set_rest(self, ob):
        """Rest Contact: the rest-pose places of the simulated bones and of the bone colliders (armature space)
        for sim.Model.set_rest. Object colliders have no rest pose (NaN): radius + radius as before."""
        m = self.model
        if not self.colliders or not m.N or not m.rest_contact.any():
            return
        bones = ob.data.bones
        H = np.array([bones[n].head_local for n in self.sim_names], dtype=np.float64)
        T = np.array([bones[n].tail_local for n in self.sim_names], dtype=np.float64)
        pbs = ob.pose.bones
        A = np.full((len(self.colliders), 3), np.nan)
        B = A.copy()
        for j, c in enumerate(self.colliders):
            if c['i'] < 0:
                continue
            Mb = _m4(pbs[c['i']].bone.matrix_local)
            A[j] = Mb[:3, :3] @ (c['off'] + (0.0, c['t0'] * c['len'], 0.0)) + Mb[:3, 3]
            B[j] = A[j] if c['shape'] == 'SPHERE' else \
                Mb[:3, :3] @ (c['off'] + (0.0, c['t1'] * c['len'], 0.0)) + Mb[:3, 3]
        m.set_rest(T, 0.5 * (H + T), A, B)

    def set_rest_spheres(self, ob, chains, index):
        """Rest Contact, rest spheres. A chain bone that lies on the body in the rest pose (chain dict 'contacts':
        per bone None or (body bone name, normal from the body to the bone tail, armature space)) gets a sphere
        that rides on that body bone and touches the bone's tail in the rest pose from the body's side: centre
        = rest tail - normal x radius. The tail stays out of it (sim._rest_spheres). The radius is the length of
        the chain (REST_SPHERE): flat enough to stand for a back or a shoulder under the strand, round enough
        to let go when the strand hangs somewhere else (a deep bow)."""
        self.rs_idx = np.zeros(0, dtype=np.int64)
        rows, anchors, cloc, rad = [], [], [], []
        bones = ob.data.bones
        i = 0
        for c in chains:
            cc = c.get('contacts')
            R = REST_SPHERE * sum(bones[n].length for n in c['bones'])
            for j, n in enumerate(c['bones']):
                k = cc[j] if cc and j < len(cc) else None
                if k is not None and k[0] in index and R > 0.0:
                    nrm = np.array(k[1], dtype=np.float64)
                    nrm /= max(float(np.linalg.norm(nrm)), 1e-300)
                    Mi = np.linalg.inv(_m4(bones[k[0]].matrix_local))
                    ctr = np.array(bones[n].tail_local, dtype=np.float64) - nrm * R
                    rows.append(i)
                    anchors.append(index[k[0]])
                    cloc.append(Mi[:3, :3] @ ctr + Mi[:3, 3])
                    rad.append(R)
                i += 1
        if rows:
            self.rs_idx = np.array(rows, dtype=np.int64)
            self.rs_anchor = np.array(anchors, dtype=np.int64)
            self.rs_cloc = np.array(cloc)
            self.rs_rad = np.array(rad)

    def rest_spheres(self, pbs, mw):
        """World centres (N,3) and radii (N,) of the rest spheres on the current pose, or (None, None)."""
        if not len(self.rs_idx):
            return None, None
        N = self.model.N
        C = np.zeros((N, 3))
        R = np.zeros(N)
        for a in np.unique(self.rs_anchor):
            k = self.rs_anchor == a
            Mb = mw @ _m4(pbs[int(a)].matrix)
            C[self.rs_idx[k]] = self.rs_cloc[k] @ Mb[:3, :3].T + Mb[:3, 3]
            R[self.rs_idx[k]] = self.rs_rad[k] * abs(np.linalg.det(Mb[:3, :3])) ** (1.0 / 3.0)
        return C, R

    # ------------------------------------------------------------------------------------------------ reading

    def adopt(self, old):
        """Keep the simulation state of the previous runtime when only parameters changed (same bones in the
        same order): a value edited during playback does not restart the swing."""
        # same bones by index (not by name: a renamed bone keeps its swing, spec 5.8 rename_live)
        same = (np.array_equal(old.sim_idx, self.sim_idx) and old.model.N == self.model.N
                and old.model.n_ext == self.model.n_ext and getattr(old, 'state', None) is not None
                and list(getattr(old, 'chain_ids', ())) == list(getattr(self, 'chain_ids', ())))
        if same:
            self.state = old.state
            self.last_frame = old.last_frame
            self.last_q = None
            self.last_written = old.last_written
        self.sim_ms = old.sim_ms
        self.timings = old.timings
        self.errors = old.errors
        self.error = old.error
        return same

    def median_ms(self):
        return statistics.median(self.sim_ms) if self.sim_ms else None

    def read(self, scene, ob_eval, depsgraph=None):
        pbs = ob_eval.pose.bones
        pbs.foreach_get("matrix_basis", self.buf16)
        banim = self.buf16.reshape(-1, 4, 4).transpose(0, 2, 1)[self.sim_idx].astype(np.float64)
        pbs.foreach_get("rotation_quaternion", self.buf4)
        qa = self.buf4.reshape(-1, 4)[self.sim_idx].astype(np.float64)
        mw = _m4(ob_eval.matrix_world)
        ext = np.empty((len(self.ext_keys), 4, 4))
        for e, ek in enumerate(self.ext_keys):
            ext[e] = mw if ek < 0 else mw @ _m4(pbs[ek].matrix)
        scale = abs(np.linalg.det(mw[:3, :3])) ** (1.0 / 3.0)
        nc = len(self.colliders)
        A = np.empty((nc, 3))
        B = np.empty((nc, 3))
        R = np.empty(nc)
        for j, c in enumerate(self.colliders):
            if c['i'] < 0:                            # object collider (sphere)
                o = self.collider_object(ob_eval, c)
                if o is None:                         # object deleted: off (never a point at the origin)
                    A[j] = B[j] = 0.0
                    R[j] = COLLIDER_OFF
                    continue
                if depsgraph is not None:
                    o = o.evaluated_get(depsgraph)
                Mo = _m4(o.matrix_world)
                A[j] = B[j] = Mo[:3, :3] @ c['off'] + Mo[:3, 3]
                R[j] = c['r'] * abs(np.linalg.det(Mo[:3, :3])) ** (1.0 / 3.0)
                continue
            Mb = mw @ _m4(pbs[c['i']].matrix)
            a = c['off'] + (0.0, c['t0'] * c['len'], 0.0)
            A[j] = Mb[:3, :3] @ a + Mb[:3, 3]
            if c['shape'] == 'SPHERE':
                B[j] = A[j]
            else:
                b = c['off'] + (0.0, c['t1'] * c['len'], 0.0)
                B[j] = Mb[:3, :3] @ b + Mb[:3, 3]
            R[j] = c['r'] * abs(np.linalg.det(Mb[:3, :3])) ** (1.0 / 3.0)
        fps = scene.render.fps / scene.render.fps_base
        g = self.gravity_vector / max(scene.unit_settings.scale_length, 1e-9)
        rs_c, rs_r = self.rest_spheres(pbs, mw)
        return sim.Frame(ext, banim, qa, A, B, R, scale=scale, gravity=g, fps=fps, time_scale=self.time_scale,
                         rs_c=rs_c, rs_r=rs_r)

    @staticmethod
    def collider_object(ob_eval, c):
        """The object of an object collider: the pointer property of the stored collider (the original object),
        or by name for colliders attached by code."""
        if c.get('ci') is not None:
            cols = ob_eval.original.jigglekit.colliders
            return cols[c['ci']].object if c['ci'] < len(cols) else None
        return bpy.data.objects.get(c['obj'])

    # ------------------------------------------------------------------------------------------------ writing

    def write(self, ob, q):
        pbs = ob.pose.bones
        pbs.foreach_get("rotation_quaternion", self.buf4)
        Q = self.buf4.reshape(-1, 4)
        Q[self.sim_idx] = q
        pbs.foreach_set("rotation_quaternion", self.buf4)
        self.last_written = Q[self.sim_idx]            # fancy index: a copy
        tag(ob, self.tag)

    def write_rest(self, ob):
        t0 = time.perf_counter()
        self.write(ob, self.rest_q)
        self.pre_times.append(time.perf_counter() - t0)

    # --------------------------------------------------------------------------------------------- frame step

    def frame(self, scene, depsgraph, ob):
        """Spec 5.5 step 3: same frame again -> the stored result; 1..max_catchup frames forward -> one step;
        any other jump -> the cache if it holds this frame with the same inputs, else a restart on the animated
        pose (always a restart at or before the scene start, except while baking)."""
        f = scene.frame_current
        sub = scene.frame_subframe
        if sub != 0.0 and not self.recording:
            # A sub-frame evaluation (motion blur, frame_set(f, subframe=...)): no step, no restart, no cache.
            # The result of the neighbouring integer frames is written (blended when both are known), so the
            # swing state is untouched and the next integer frame continues it.
            q = self.subframe_q(f, sub)
            if q is not None:
                if getattr(depsgraph, 'mode', 'VIEWPORT') == 'RENDER':
                    flush_animation(ob, ob.evaluated_get(depsgraph))
                self.write(ob, q)
            return
        t0 = time.perf_counter()
        fr = self.read(scene, ob.evaluated_get(depsgraph), depsgraph)
        sig = signature(fr) if (self.use_cache or self.last_q is not None) else None
        if self.cache_anim != ANIM[0]:
            self.cache.clear()
            self.cache_anim = ANIM[0]
        if (fr.fps < 6.0 or fr.time_scale > 3.0) and LOW_FPS_NOTE not in self.notes:
            self.notes.append(LOW_FPS_NOTE)          # spec 7: stable, but the swing is coarse
        t1 = time.perf_counter()
        last = self.last_frame
        restored = False
        if last is not None and f == last and self.last_q is not None and sig == self.last_sig:
            q = self.last_q
            restored = True
        elif last is not None and 1 <= f - last <= self.max_catchup and (self.recording or f > scene.frame_start):
            q = sim.step(self.model, self.state, fr, n_frames=f - last)
        else:
            ent = None
            if self.use_cache and not self.recording and f > scene.frame_start:
                ent = self.cache.get(f)
            if ent is not None and ent[0] == sig:
                q = self.restore(ent, fr)
                restored = True
                self.cache_hits += 1
            else:
                q = sim.reset(self.model, self.state, fr)
                self.resets += 1
                if not self.root_checked:
                    self.check_roots(fr)
            self.hist.clear()                         # a jump: the neighbours of this frame are not known
        t2 = time.perf_counter()
        if not restored:
            self.contacts_total += self.state.contacts
            if self.recording:
                self.record[f] = np.array(q, dtype=np.float64)
            elif self.use_cache:
                self.store(f, sig, q)
        if getattr(depsgraph, 'mode', 'VIEWPORT') == 'RENDER':
            flush_animation(ob, ob.evaluated_get(depsgraph))
        self.write(ob, q)
        t3 = time.perf_counter()
        self.hist[f] = np.array(q, dtype=np.float64)
        while len(self.hist) > HIST_FRAMES:
            del self.hist[min(self.hist)]
        self.last_frame = f
        self.last_q = q
        self.last_sig = sig
        self.timings.append((t1 - t0, t2 - t1, t3 - t2))
        self.sim_ms.append(1000.0 * (t2 - t1))

    def subframe_q(self, f, sub):
        """Rotations for the time f + sub (0 < sub < 1) from the computed integer frames: a normalised blend of
        frames f and f + 1 when both are known, else the one that is known; None when neither is."""
        a, b = self.hist.get(f), self.hist.get(f + 1)
        if a is None or b is None:
            return a if a is not None else b
        b = np.where((np.einsum('nj,nj->n', a, b) < 0.0)[:, None], -b, b)
        q = a + (b - a) * sub
        return q / np.maximum(np.sqrt(np.einsum('nj,nj->n', q, q)), 1e-300)[:, None]

    def rewrite(self, scene, ob):
        """Put this frame's result back into the pose. For an evaluation that does not hold the armature (a view
        layer that excludes it): frame_change_pre has no depsgraph, so it wrote the rest rotations for that
        view layer too, and nothing would follow."""
        f, sub = scene.frame_current, scene.frame_subframe
        if sub != 0.0:
            q = self.subframe_q(f, sub)
        else:
            q = self.last_q if self.last_frame == f else None
        if q is not None:
            self.write(ob, q)

    def check_roots(self, fr):
        """Spec 7: a chain root inside a collider gets a note in Result (the particle may stick; nothing fails)."""
        self.root_checked = True
        m = self.model
        if not len(fr.col_r) or not m.N:
            return
        from . import collide
        roots = m.chain_start
        rad = fr.col_r[None, :]
        dh = collide.distance(self.state.H[roots], fr.col_a, fr.col_b)          # (chains, colliders)
        dt_ = collide.distance(self.state.T[roots], fr.col_a, fr.col_b)
        # head and rest tail both inside: the particle is pushed out of its rest pose at once
        inside = (dh < rad) & (dt_ < rad + (m.radius[roots] * fr.scale)[:, None]) & m.collide[roots][:, None] \
            & ~m.rest_contact[roots][:, None]              # Rest Contact: a bone is never pushed out of its rest pose
        for ci in np.flatnonzero(inside.any(axis=1)):
            name = self.sim_names[int(m.chain_start[ci])]
            msg = "Chain root %s is inside a collider; the bone may stick out of its pose. Move or shrink the collider." % name
            if msg not in self.notes:
                self.notes.append(msg)

    # ---------------------------------------------------------------------------------------------------- cache

    def store(self, f, sig, q):
        st = self.state
        c = self.cache
        if f in c:
            del c[f]                                    # re-insert: newest last
        c[f] = (sig, np.asarray(q, dtype=np.float32), st.P.astype(np.float32), st.V.astype(np.float32),
                st.Tprev.astype(np.float32))
        while len(c) > self.cache_max:
            del c[next(iter(c))]                        # oldest first

    def restore(self, ent, fr):
        """Pose and swing state of a cached frame (float32 copy): playing on from it continues the swing."""
        _sig, q, P, V, T = ent
        st = self.state
        st.P = P.astype(np.float64)
        st.V = V.astype(np.float64)
        st.Tprev = T.astype(np.float64)
        st.T = st.Tprev.copy()
        st.ext_R = fr.ext[:, :3, :3].copy()
        st.ext_t = fr.ext[:, :3, 3].copy()
        st.col_a, st.col_b = fr.col_a.copy(), fr.col_b.copy()
        st.rs_c = None if fr.rs_c is None else fr.rs_c.copy()
        st.q_prev = q.astype(np.float64)
        st.valid = True
        st.contacts = 0
        return st.q_prev.copy()


def flush_animation(ob, ob_eval):
    """Render only (spec 5.5 render_applies, measured on 4.2 / 4.5 / 5.2): a render depsgraph does not copy the
    animated pose back to the original armature, and the DATA tag after our write makes it copy the original
    (un-animated) pose again, so every other animated bone would lose its animation in the rendered frame. The
    evaluated pose (animation, NLA, drivers) is copied to the original first, as the viewport depsgraph does."""
    src, dst = ob_eval.pose.bones, ob.pose.bones
    n = len(dst)
    for attr, k in (("location", 3), ("rotation_quaternion", 4), ("rotation_euler", 3),
                    ("rotation_axis_angle", 4), ("scale", 3)):
        buf = np.empty(n * k, dtype=np.float32)
        src.foreach_get(attr, buf)
        dst.foreach_set(attr, buf)


def signature(fr):
    """Hash of the inputs a frame uses (spec 5.5 step 2): armature and external parent matrices, bone bases,
    collider points and radii, rest spheres, gravity, scale, fps, time scale; rounded to 1e-4."""
    x = np.concatenate((fr.ext.ravel(), fr.banim.ravel(), fr.col_a.ravel(), fr.col_b.ravel(), fr.col_r,
                        fr.gravity, (fr.scale, fr.fps, fr.time_scale)))
    if fr.rs_r is not None:
        x = np.concatenate((x, fr.rs_c.ravel(), fr.rs_r))
    return hash(np.round(x * 1e4).astype(np.int64).tobytes())


def tag(ob, how):
    if how == 'DATA':
        ob.update_tag(refresh={'DATA'})
    elif how == 'OBJECT':
        ob.update_tag(refresh={'OBJECT'})
    elif how == 'TIME':
        ob.update_tag(refresh={'TIME'})
    elif how == 'ALL':
        ob.update_tag()
    # 'NONE': no tag (control in the probes)


def attach(ob, chains, colliders=(), **kw):
    rt = ArmRT(ob, chains, colliders, **kw)
    RT[rt.key] = rt
    invalidate()
    return rt


def reset_all():
    for rt in RT.values():
        rt.state.valid = False
        rt.last_frame = None
        rt.last_q = None
        rt.cache.clear()
        rt.hist.clear()


def anim_changed():
    """An Action changed (keys edited): every cache is stale (spec 5.5, anim_version)."""
    ANIM[0] += 1


# ------------------------------------------------------------------------------- stored data and handler loop

def restore_rest(ob, names):
    """Write rest_q to chain bones that are no longer simulated (chain off, cut, removed), once."""
    pbs = ob.pose.bones
    done = False
    for n in names:
        pb = pbs.get(n)
        if pb is not None and pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = tuple(pb.jigglekit.rest_q)
            done = True
    if done:
        tag(ob, 'DATA')


def stale(rt, ob):
    """Rebuild needed: parameter version, other object behind the name, bone count or names by cached index."""
    if rt.built_version != ob.jigglekit.version or rt.ptr != ob.as_pointer() or len(ob.pose.bones) != rt.n_pose:
        return True
    pbs = ob.pose.bones
    for i, n in zip(rt.sim_idx.tolist(), rt.sim_names):
        if pbs[i].name != n:
            return True
    return False


def written(ob):
    """{bone name: quaternion the live simulation wrote last} of an armature (empty when it is not simulated)."""
    rt = RT.get(key(ob))
    if rt is None or rt.last_written is None:
        return {}
    return {n: rt.last_written[i].astype(np.float64) for i, n in enumerate(rt.sim_names)}


def restore_all():
    """Every simulated bone back to its rest rotation (the add-on is being disabled: without this the bones
    would stay in the middle of a swing and that pose would be saved with the file)."""
    for ob in bpy.data.objects:
        if ob.type != 'ARMATURE' or ob.pose is None:
            continue
        rt = RT.get(key(ob))
        if rt is None or rt.ptr != ob.as_pointer():
            continue
        try:
            rt.write(ob, rt.rest_q)
        except Exception as e:  # noqa: BLE001
            print("JiggleKit: restore failed on %s: %r" % (ob.name, e))


def live(ob):
    jk = ob.jigglekit
    return (jk.enabled or key(ob) in BAKING) and not jk.use_baked and len(jk.chains) > 0


def ensure(ob):
    """Runtime of an armature with stored chains, rebuilt when stale; None when the armature is not live."""
    from . import model
    rt = RT.get(key(ob))
    if rt is not None and not rt.from_props:
        return rt                                  # attached from code (tests)
    if not live(ob):
        if rt is not None:
            RT.pop(key(ob), None)
            restore_rest(ob, rt.sim_names)
        return None
    if rt is None or stale(rt, ob):
        rt = model.rebuild(ob)
    return rt


def rebuild(ob):
    from . import model
    return model.rebuild(ob)


def rebuild_all():
    """After undo / redo / load: the runtime is dropped and rebuilt from stored data (code-attached rigs, used by
    the speed tests, are only reset)."""
    reset_all()
    invalidate()
    arms = {key(ob): ob for ob in bpy.data.objects if ob.type == 'ARMATURE'}
    for k in list(RT):
        if k not in arms or RT[k].from_props:
            RT.pop(k, None)
    for ob in arms.values():
        if live(ob):
            try:
                rebuild(ob)
            except Exception as e:  # noqa: BLE001
                print("JiggleKit: rebuild failed:", repr(e))


def armatures(scene):
    """Armatures of the scene that have chains or a runtime. The list is cached per scene (walking
    scene.objects twice per frame costs more than the simulation in a large scene) and rebuilt when the number
    of objects changes, on invalidate() (chains added or removed, parameter edits, undo, load), when a cached
    object is gone, and at least every RESCAN_CALLS calls."""
    ent = _ARMS.get(scene.name)
    n = len(scene.objects)
    if ent is not None and ent[0] == n and ent[1] < RESCAN_CALLS:
        ent[1] += 1
        try:
            for ob in ent[2]:
                ob.name                              # ReferenceError: the object was removed
            return ent[2]
        except ReferenceError:
            pass
    SCANS[0] += 1
    obs = [ob for ob in scene.objects
           if ob.type == 'ARMATURE' and (len(ob.jigglekit.chains) or key(ob) in RT)]
    _ARMS[scene.name] = [n, 0, obs]
    return obs


def in_graph(ob, depsgraph):
    """False when the depsgraph does not hold the object (a view layer that excludes its collection): such an
    evaluation must not step or restart the simulation. Blender calls the frame handlers once per view layer
    of the scene (frame_change_pre without a depsgraph, measured on 4.2 / 4.5 / 5.2)."""
    if depsgraph is None:
        return True
    try:
        return bool(ob.evaluated_get(depsgraph).is_evaluated)
    except Exception:  # noqa: BLE001
        return True


def active(scene, depsgraph=None):
    for ob in list(armatures(scene)):
        rt = RT.get(key(ob))
        if rt is None and not len(ob.jigglekit.chains):
            continue
        if not in_graph(ob, depsgraph):
            continue
        try:
            rt = ensure(ob)
        except Exception as e:  # noqa: BLE001
            print("JiggleKit: rebuild failed:", repr(e))
            continue
        if rt is None or not rt.enabled:
            continue
        yield ob, rt


def _failed(rt, ob, scene, e):
    rt.errors += 1
    if not rt.error:                     # first message only (Result panel); printed once
        rt.error = "%s: %s" % (type(e).__name__, e)
        print("JiggleKit: simulation error on %s (frame %d): %r" % (ob.name, scene.frame_current, e))
    rt.state.valid = False               # next frame restarts on the animated pose
    rt.last_frame = None
    rt.last_q = None
    rt.hist.clear()


def frame_pre(scene, depsgraph=None):
    STATS['pre_calls'] += 1
    try:
        for ob, rt in active(scene, depsgraph):
            if rt.busy:
                continue
            try:
                rt.write_rest(ob)
            except Exception as e:  # noqa: BLE001  never break the handler (spec 5.5, step 0)
                _failed(rt, ob, scene, e)
    except Exception as e:  # noqa: BLE001
        print("JiggleKit: frame_change_pre failed:", repr(e))


def frame_post(scene, depsgraph):
    STATS['post_calls'] += 1
    any_rt = False
    for ob, rt in active(scene):
        if rt.busy:
            continue
        any_rt = True
        rt.busy = True
        try:
            if in_graph(ob, depsgraph):
                rt.frame(scene, depsgraph, ob)
            else:
                rt.rewrite(scene, ob)
        except Exception as e:  # noqa: BLE001  never break the handler (spec 5.5, step 0)
            _failed(rt, ob, scene, e)
        finally:
            rt.busy = False
    if any_rt:
        redraw_panels()


def redraw_panels(min_interval=0.25):
    """Status line refresh during playback, at most 4 times a second."""
    now = time.perf_counter()
    if now - _REDRAW[0] < min_interval:
        return
    _REDRAW[0] = now
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'UI':
                        reg.tag_redraw()


def status(ob):
    """(text, alert, slow) for the panel status line (spec 4.1 + Pavel's request)."""
    jk = ob.jigglekit
    if jk.use_baked:
        return "Baked (live paused)", False, False
    if not jk.enabled or not len(jk.chains):
        return "Off", False, False
    rt = RT.get(key(ob))
    med = rt.median_ms() if rt is not None else None
    if med is None:
        return "Live", False, False
    slow = med > SLOW_MS
    return "Live \u00b7 %.1f ms/frame" % med, slow, slow

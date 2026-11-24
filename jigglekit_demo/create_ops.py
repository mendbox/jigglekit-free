# SPDX-License-Identifier: GPL-3.0-or-later
""""One click" (spec 5.7): Create Jiggle from Selection. Stages 1-3 (read, region geometry, weights) compute with
numpy without writing anything; stages 4-6 (bones, weights, finish) write after all checks and roll back on error.

Settings come from Scene.jigglekit_ui (the Create panel); an operator property set explicitly by the caller
(scripts, tests, the redo panel) wins. Chains are stored in Object.jigglekit.chains, bones carry
PoseBone.jigglekit; the live simulation starts through model.rebuild().

Also here: add_chains_from_bones (chains on existing bones, no weights) and paint_weights (Weights for Selection:
the same weight path for a second mesh, e.g. clothing)."""
import json
import re
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Quaternion, Vector, kdtree

from . import bake, ids, model, presets, props, regions, weights
from .const import DEMO_BONE_LIMIT, EDITION, RESULT

COLLECTION = "JiggleKit"
MIRROR_TOL = 3e-3            # x D (mesh bbox diagonal)
MIRROR_MIN_MATCH = 0.70
MIRROR_LOOSE_EDGES = 1.5     # second try on a mesh that is not exactly symmetric: nearest vertex within this many
                             # edge lengths (95th percentile in the selection) of the mirrored position
MIRROR_LOOSE_R = 0.35        # ... and never further than this share of the region radius
BAND_MIN_EDGES = 5.0         # weight ramp starts at least this many edges wide (95th percentile length in the region)
MAX_EDGE_STEP = 0.30         # largest mask step across one edge after widening (check limit 0.35)
ATT_TWIST_DEG = 20.0         # sideways root swing the chain attachment ramp is sized for (spec 5.7 tear criterion)
ATT_MAX_SHARE = 0.75         # attachment ramp covers at most this share of the chain
WIDE_WARNING = ("The area is wide for one chain: the top may stretch when it swings sideways. For a skirt or cape, "
                "make 2-3 chains side by side.")


TIP_WARNING = ("The part of this mesh that is not selected is at the tip of the chain, not at its root: it keeps its "
               "old weights. For a whole hair or tail, select all of it.")


class CreateError(Exception):
    pass


def group_sums(me, vg_names, bones, idx):
    """{bone name: summed weight of the vertices idx}, only for the groups named after a bone in `bones`."""
    acc = {}
    for gw in read_groups(me, idx).values():
        for g, w in gw.items():
            nm = vg_names[g] if g < len(vg_names) else None
            if nm in bones:
                acc[nm] = acc.get(nm, 0.0) + w
    return acc


def too_large(limit_ratio):
    return "The selection covers more than %d%% of the mesh." % round(100 * limit_ratio)


def whole_part_error(limit_ratio, sums, what="chain", active_hint=True):
    """Message for a selection of most of a mesh that is not one part hanging from one line of bones."""
    why = ("the mesh has no weights for the bones of this armature" if not any(w > 0.0 for w in sums.values())
           else "its weights are on bones that branch (a body, not one strand)")
    return ("%s That is fine for a whole separate part (hair, tail), but %s, so it is not clear where the %s "
            "attaches. Select a smaller area%s."
            % (too_large(limit_ratio), why, what,
               ", or set Parent to Active Bone and make the bone it hangs from the active bone"
               if active_hint else ""))


new_id = ids.new_id


def _np(m):
    return np.array(m, dtype=np.float64)


def write_result(scene, res):
    try:
        setattr(scene, RESULT, json.dumps(res))
    except (AttributeError, TypeError):
        scene[RESULT] = json.dumps(res)


def show_collection(coll, msgs):
    """The bones of a chain must be seen: un-hide the JiggleKit bone collection (and its parents). When another
    collection is in Solo the bones still do not show: say where they are."""
    if coll is None:
        return
    try:
        c = coll
        while c is not None:
            if not c.is_visible:
                c.is_visible = True
            c = c.parent
    except (AttributeError, RuntimeError):          # linked armature data
        pass
    if not getattr(coll, "is_visible_effective", True):
        m = ('The chain bones are in the bone collection "%s", which is hidden now (another collection is in Solo, '
             'or the armature is linked). See Armature properties > Bone Collections.' % coll.name)
        if m not in msgs:
            msgs.append(m)


def find_armature(ob, name):
    """(armature object, skinned) from the panel field or the first enabled Armature modifier."""
    mods = [m for m in ob.modifiers if m.type == 'ARMATURE' and m.object is not None and m.show_viewport]
    if name:
        arm = bpy.data.objects.get(name)
        if arm is None or arm.type != 'ARMATURE':
            return None, False
        return arm, any(m.object == arm for m in mods)
    if mods:
        return mods[0].object, True
    return None, False


# ------------------------------------------------------------------------------------------- stages 1-3: plan

def read_mesh(ob):
    ob.update_from_editmode()
    me = ob.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    nr = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("normal", nr)
    sel = np.zeros(nv, dtype=bool)
    me.vertices.foreach_get("select", sel)
    ed = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ed)
    hid = np.zeros(nv, dtype=bool)
    me.vertices.foreach_get("hide", hid)
    return (co.reshape(-1, 3).astype(np.float64), nr.reshape(-1, 3).astype(np.float64), sel & ~hid,
            ed.reshape(-1, 2).astype(np.int64))


def read_groups(me, idx):
    """{vertex: {group index: weight}} for the given vertices (only these are touched, spec 5.7 stage 1)."""
    vs = me.vertices
    return {int(i): {g.group: g.weight for g in vs[int(i)].groups} for i in idx}


def mirror_pairs(A, sel_idx, tol):
    """Index of the mirrored vertex (x -> -x) for each selected vertex, -1 if none within the tolerance."""
    kd = kdtree.KDTree(len(A))
    for i, p in enumerate(A):
        kd.insert(Vector(p), i)
    kd.balance()
    out = np.full(len(sel_idx), -1, dtype=np.int64)
    for j, i in enumerate(sel_idx):
        p = A[i]
        _co, k, d = kd.find(Vector((-p[0], p[1], p[2])))
        if k is not None and d <= tol:
            out[j] = k
    return out


def mirror_selection(A, edges, sel, D, R):
    """Mirrored selection mask (x -> -x) or None, and exact (bool).

    First the exact match (MIRROR_TOL x D). A garment or a sculpt is often a few millimetres off symmetric: then
    the nearest vertex within MIRROR_LOOSE_EDGES edge lengths of the mirrored position counts, vertices enclosed
    by matched ones are added (two selected vertices may land on the same mirrored one and skip its neighbour),
    and vertices of the selection itself are left out. The bone is mirrored through the armature's X plane either
    way; the weights are computed per side from the side's own vertices."""
    nv = len(A)
    sel_idx = np.flatnonzero(sel)
    pairs = mirror_pairs(A, sel_idx, MIRROR_TOL * D)
    ok = pairs >= 0
    if ok.mean() >= MIRROR_MIN_MATCH:
        msel = np.zeros(nv, dtype=bool)
        msel[pairs[ok]] = True
        return msel, True
    es = edges[sel[edges[:, 0]] & sel[edges[:, 1]]]
    if not len(es):
        return None, False
    e95 = float(np.percentile(np.linalg.norm(A[es[:, 1]] - A[es[:, 0]], axis=1), 95))
    tol = max(MIRROR_TOL * D, min(MIRROR_LOOSE_EDGES * e95, MIRROR_LOOSE_R * R))
    pairs = mirror_pairs(A, sel_idx, tol)
    ok = pairs >= 0
    if ok.mean() < MIRROR_MIN_MATCH:
        return None, False
    msel = np.zeros(nv, dtype=bool)
    msel[pairs[ok]] = True
    deg = np.bincount(edges.ravel(), minlength=nv)
    inn = np.bincount(np.concatenate((edges[msel[edges[:, 1]], 0], edges[msel[edges[:, 0]], 1])), minlength=nv)
    msel |= (deg > 0) & (inn == deg)                 # enclosed by matched vertices
    msel &= ~sel
    if int(msel.sum()) < 4:
        return None, False
    return msel, False


def base_mesh_warnings(ob):
    """Spec 7: weights are computed on the base mesh (rest coordinates). Say so when the shown shape differs."""
    out = []
    if ob.data.shape_keys is not None and len(ob.data.shape_keys.key_blocks) > 1:
        out.append("The mesh has shape keys: the area is measured on the base shape.")
    before = []
    for m in ob.modifiers:
        if m.type == 'ARMATURE':
            break
        if m.show_viewport:
            before.append(m.name)
    else:
        before = []                                   # no Armature modifier: nothing comes "before" it
    if before:
        out.append("Modifiers before Armature (%s) change the mesh; the area is measured on the base mesh."
                   % ", ".join(before[:3]))
    return out


def plan_create(ob, s):
    """Stages 1-3. s: settings (operator). Returns a plan dict; raises CreateError. Writes nothing."""
    t0 = time.perf_counter()
    warn = []
    arm, skinned = find_armature(ob, s.armature)
    if arm is None:
        raise CreateError("No armature found. Add an Armature modifier or pick an armature in the panel.")
    if len(arm.data.bones) == 0:
        raise CreateError("The armature has no bones.")
    if model.bones_locked(arm):
        raise CreateError(model.LINKED_BONES % arm.name)
    warn.extend(base_mesh_warnings(ob))
    co, nr, sel, edges = read_mesh(ob)
    nv = len(co)
    nsel = int(sel.sum())
    if nsel < 4:
        raise CreateError("Select at least 4 vertices.")
    over = nsel > s.limit_ratio * nv
    if over and presets.get(s.region)['kind'] != 'CHAIN':
        # one bone for most of a mesh is never what was meant. A chain region may be a whole separate part
        # (a hair or tail mesh): checked below, when the weights are known
        raise CreateError(too_large(s.limit_ratio) + " Select a smaller area. For a whole separate part (hair, "
                          "tail, accessory) choose a chain region: Ears, Hair Strand, Tail, Cloth Flap or Accessory.")
    me_ = ob.data
    lv = np.empty(len(me_.loops), dtype=np.int32)
    me_.loops.foreach_get("vertex_index", lv)
    lt = np.empty(len(me_.polygons), dtype=np.int32)
    me_.polygons.foreach_get("loop_total", lt)
    ls = np.empty(len(me_.polygons), dtype=np.int32)
    me_.polygons.foreach_get("loop_start", ls)
    lp = np.empty(len(lv), dtype=np.int64)
    for_all = np.repeat(np.arange(len(lt)), lt)
    lp[:] = -1
    if len(lt) and np.array_equal(ls, np.concatenate(([0], np.cumsum(lt)[:-1]))):
        lp[:len(for_all)] = for_all                       # loops are stored face after face (the normal layout)
    else:
        for i, (a, n) in enumerate(zip(ls.tolist(), lt.tolist())):
            lp[a:a + n] = i
    on_face = lp >= 0
    n_parts = weights.count_parts(nv, edges, sel, weights.face_pairs(lv[on_face], lp[on_face], sel))
    if n_parts > 1:
        # never one chain (or one bone) across separate patches: the bones would run through the body between
        # them (GUI session 2026-11-23: a tail chain that started at the chest)
        raise CreateError("Selection has %d separate parts: select one part." % n_parts)
    for o in (ob, arm):
        sc = np.array(o.matrix_world.to_scale())
        if np.abs(sc - 1.0).max() > 1e-3:
            msg = ("Non-uniform scale: results may be off." if np.ptp(sc) > 1e-3 * sc.max()
                   else "Object scale is not applied; results may be off.")
            if msg not in warn:
                warn.append(msg)
    Mx = _np(arm.matrix_world.inverted()) @ _np(ob.matrix_world)
    A = co @ Mx[:3, :3].T + Mx[:3, 3]
    Nn = nr @ np.linalg.inv(Mx[:3, :3])          # normal matrix = inverse transpose
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1), 1e-30)[:, None]
    D = float(np.linalg.norm(A.max(axis=0) - A.min(axis=0)))
    mesh_center = A.mean(axis=0)

    pre = presets.get(s.region)
    kind = pre['kind']
    nb = 1 if kind == 'COMPACT' else (s.segments if s.segments > 0 else pre['bones'])
    use_mirror = pre['mirror'] if s.mirror == 'PRESET' else s.mirror == 'ON'
    base = (s.name.strip() or presets.LABELS[s.region])[:40]
    if EDITION == 'DEMO':
        have = model.demo_bones_in_file()          # spec 9.3: the limit is for the whole file
        if have + nb * (2 if use_mirror else 1) > DEMO_BONE_LIMIT:
            raise CreateError("Demo limit: %d jiggle bones. Get the full version." % DEMO_BONE_LIMIT)

    # groups <-> bones
    me = ob.data
    vg_names = [vg.name for vg in ob.vertex_groups]
    vg_locked = [vg.lock_weight for vg in ob.vertex_groups]
    jk_bones = {pb.name for pb in arm.pose.bones if pb.jigglekit.chain_id}
    bone_segs = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm.data.bones
                 if b.name not in jk_bones}
    deform_bones = {b.name for b in arm.data.bones if b.use_deform}
    if not bone_segs:
        raise CreateError("The armature has no bones.")

    def sums_of(idx):
        return group_sums(me, vg_names, bone_segs, idx)

    active = None
    if s.parent_mode == 'ACTIVE_BONE' and arm.data.bones.active is not None \
            and arm.data.bones.active.name in bone_segs:
        active = arm.data.bones.active.name

    def parent_for(idx, point, sums=None):
        if active is not None:
            return active, 'ACTIVE'
        return regions.pick_parent(sums_of(idx) if sums is None else sums, bone_segs, point)

    G = weights.graph(nv, edges, A)

    def ring(mask_):
        """mask_ plus its one-edge neighbours."""
        out = mask_.copy()
        e = edges
        out[e[mask_[e[:, 0]], 1]] = True
        out[e[mask_[e[:, 1]], 0]] = True
        return out

    sides = []
    sel_idx = np.flatnonzero(sel)
    # first side: geometry from the selection
    if kind == 'COMPACT':
        # which side is "out": away from the deform skeleton under the selection (the closest points of the
        # bones that hold its weights), not the vertex normals alone (they point inwards on some garments)
        gs = {n: w for n, w in sums_of(sel_idx).items() if w > 0.0 and n in deform_bones}
        inside = regions.closest_on_segments(A[sel].mean(axis=0), [bone_segs[n] for n in gs], list(gs.values())) \
            if gs else None
        geo = regions.compact(A[sel], Nn[sel], mesh_center, inside)
        if geo['guessed']:
            warn.append("Direction guessed from shape.")
        if geo['flipped']:
            warn.append("The normals of the selected area point inwards; the bone was pointed outwards, away from "
                        "the skeleton.")
        J = np.array([geo['head'], geo['tail']])
        if np.linalg.norm(J[1] - J[0]) < 1e-4 * D:
            raise CreateError("The region is too small or too thin.")
        parent, how = parent_for(sel_idx, geo['head'])
        R = geo['R']
        center = geo['c']
    else:
        B = weights.border(edges, sel)
        sums = sums_of(sel_idx)
        par0, _h0 = parent_for(sel_idx, A[sel].mean(axis=0), sums)

        def axis_from(base_):
            """Main axis from a base set (where the part meets the body) to the far end of the region, and the
            distance between the two; None when there is no direction."""
            d_in = weights.dijkstra(G, base_, allowed=sel)
            fin = sel & np.isfinite(d_in)
            far = fin & (d_in >= 0.8 * float(np.max(d_in[fin])))
            v = A[far].mean(axis=0) - A[base_].mean(axis=0)
            ln = float(np.linalg.norm(v))
            return (v / ln, ln) if far.any() and ln > 1e-6 * D else None

        # The base of the chain is the selection border (where the part meets the rest of the mesh). A selection
        # of a whole mesh island has no border, and one of most of a mesh only a stray one: a separate hair, tail
        # or accessory mesh. Then the base is the end that sits on the bone that carries the part: the root of the
        # line of bones the part is weighted to (regions.weight_line), or the active bone when Parent says so.
        # A selection under the limit that has a border is never touched by this.
        root_set, at_border, ax = B, True, None
        if over or not B.any():
            anchor, seg = active, bone_segs.get(active)
            if anchor is None:
                share, root, on = regions.weight_line(sums, {b.name: b.parent.name if b.parent else None
                                                             for b in arm.data.bones}, bone_segs)
                if share >= regions.LINE_SHARE:
                    # one bone (hair on Head): the end nearest the bone. Several bones (hair bones, tail bones):
                    # they run inside the part, the part starts at the joint of the first one
                    anchor = root
                    seg = bone_segs[root] if len(on) == 1 else (bone_segs[root][0], bone_segs[root][0])
                elif over:
                    raise CreateError(whole_part_error(s.limit_ratio, sums))
            if anchor is not None:
                near = np.zeros(nv, dtype=bool)
                near[sel_idx[regions.base_near(A[sel], seg)]] = True
                ax = axis_from(near)
                if not B.any():
                    root_set, at_border = near, False
                elif ax is not None and float((A[B].mean(axis=0) - A[near].mean(axis=0)) @ ax[0]) >= 0.5 * ax[1]:
                    root_set, at_border = near, False        # the unselected rest is at the far end, not at the root
                    warn.append(TIP_WARNING)
        if at_border:
            ax = axis_from(B) if B.any() else None
        else:
            warn.append('Whole part: the chain starts at the end nearest the bone "%s". If that is the wrong end, set '
                        'Parent to Active Bone, make the bone it hangs from the active bone and create again.' % anchor)
        axis = ax[0] if ax is not None else None
        # bones must span at least 3 edge rings (smooth split), fewer segments otherwise
        a_est = axis if axis is not None else regions.pca(A[sel])[2][:, 2]
        es = edges[sel[edges[:, 0]] & sel[edges[:, 1]]]
        ev = A[es[:, 1]] - A[es[:, 0]]
        along = np.abs(ev @ a_est)
        axial = along[along > 0.7 * np.linalg.norm(ev, axis=1)]
        e_ax = float(np.median(axial)) if len(axial) else 0.0
        n_cap = regions.max_segments(A[sel], a_est, e_ax)
        if nb > n_cap:
            warn.append("The selected area has too few edge rings for %d bones; created %d." % (nb, n_cap))
            nb = n_cap
        try:
            geo = regions.chain(A[sel], nb, base_score=root_set[sel].astype(np.float64),
                                parent_seg=bone_segs.get(par0), mesh_size=D, axis=axis,
                                back=0.1 if at_border else 0.0)
        except regions.RegionError as e:
            raise CreateError(str(e))
        J = geo['joints']
        R = geo['R']
        center = geo['c']
        if at_border and B.any():
            # the chain must start where the part meets the rest of the mesh: a base far outside the border
            # (further than the border's own radius + half the chain) would put the first bone somewhere else
            bc = A[B].mean(axis=0)
            br = float(np.linalg.norm(A[B] - bc, axis=1).max())
            if float(np.linalg.norm(J[0] - bc)) > br + 0.5 * float(geo['length']):
                raise CreateError("The chain base is far from where the selected area meets the mesh. Select "
                                  "one connected part, from its root to its tip.")
        # parent from the base layer and its one-edge ring
        u_sel = weights.chain_param(A, J)
        base_mask = ring(sel & (u_sel <= 1.0))
        parent, how = parent_for(np.flatnonzero(base_mask), J[0])
    sides.append(dict(sel=sel, J=J, R=R, parent=parent, how=how, x=float(center[0])))

    if use_mirror:
        if abs(center[0]) < 0.05 * R:
            pass                                    # centred selection: one bone, no mirror (spec 5.7)
        else:
            msel, exact = mirror_selection(A, edges, sel, D, R)
            if msel is None:
                warn.append("Mirror: the other side of the mesh does not match this one; created one side only. "
                            "Select the other side and run Create again with Mirror X off.")
            else:
                if not exact:
                    warn.append("Mirror: the mesh is not exactly symmetric; the other side was matched by distance.")
                mJ = regions.mirror_x(J)
                midx = np.flatnonzero(msel)
                mpar, mhow = parent_for(midx, mJ[0])
                sides.append(dict(sel=msel, J=mJ, R=R, parent=mpar, how=mhow, x=-float(center[0])))
    t1 = time.perf_counter()

    # stage 3: masks and shares per side
    for sd in sides:
        side_weights(G, edges, A, sd, kind, nb, s.core, s.falloff, s.weight_max, warn)
        sd['suffix'] = regions.side_suffix(sd['x']) if len(sides) == 2 else ""
    t2 = time.perf_counter()

    up_z = _np(arm.matrix_world.inverted())[:3, :3] @ np.array([0.0, 0.0, 1.0])
    up_y = _np(arm.matrix_world.inverted())[:3, :3] @ np.array([0.0, 1.0, 0.0])
    if not skinned:
        warn.append("Mesh is not skinned to this armature (no Armature modifier). Bones were created but "
                    "weights were skipped.")
    return dict(ob=ob, arm=arm, skinned=skinned, kind=kind, nb=nb, base=base, preset=s.region, pre=pre,
                sides=sides, warn=warn, D=D, up_z=up_z, up_y=up_y, vg_names=vg_names, vg_locked=vg_locked,
                deform_bones=deform_bones, weight_mode=s.weight_mode, max_influences=s.max_influences,
                timings=dict(read_geo=t1 - t0, weights=t2 - t1))


def side_weights(G, edges, A, sd, kind, nb, core, falloff, weight_max, warn):
    """Stage 3 for one side: mask M (new weight share per vertex) and per-bone shares. sd needs sel, R, J.

    Edge smoothness is guaranteed by measurement, not by a fixed width: the weight ramp starts at least
    BAND_MIN_EDGES edges wide and is widened (keeping core : falloff) until no edge carries a mask step above
    MAX_EDGE_STEP. A ramp can still come out steeper than the plain width suggests when the selection is shallower
    than the core (peak normalisation) or the band runs over longer edges than the selection. Chains: the
    attachment ramp a(u) is long enough that a ATT_TWIST_DEG sideways swing of the root does not stretch an edge by
    more than MAX_EDGE_STEP (wide flaps), see attach_ramp()."""
    nv = len(A)
    sl = sd['sel']
    f_in, f_out = core * sd['R'], falloff * sd['R']
    eb = edges[sl[edges[:, 0]] | sl[edges[:, 1]]]
    e95 = float(np.percentile(np.linalg.norm(A[eb[:, 0]] - A[eb[:, 1]], axis=1), 95)) if len(eb) else 0.0
    span = f_in + f_out
    if 0.0 < span < BAND_MIN_EDGES * e95:
        k = BAND_MIN_EDGES * e95 / span
        f_in, f_out = f_in * k, f_out * k
    for _it in range(6):
        sdist, _B = weights.signed_distance(G, edges, sl, f_out)
        m = weights.mask(sdist, sl, f_in, f_out, weight_max)
        near = m > 0.0
        step = weights.edge_step(edges[near[edges[:, 0]] | near[edges[:, 1]]], m)
        if step <= MAX_EDGE_STEP or f_in + f_out <= 0.0:
            break
        k = 1.05 * step / MAX_EDGE_STEP
        f_in, f_out = f_in * k, f_out * k
    sd['band'] = (f_in, f_out)
    if kind == 'CHAIN':
        u, dist = weights.chain_param(A, sd['J'], with_dist=True)
        ramp, capped = attach_ramp(sd['J'], dist[sl], nb)
        if capped and WIDE_WARNING not in warn:
            warn.append(WIDE_WARNING)
        W, att = weights.chain_split(u, nb, ramp)
        M = m * att
        M[M < 1e-3] = 0.0
        sd['ramp'] = ramp
    else:
        M = m
        W = np.ones((nv, 1))
    sd.update(M=M, shares=W, m=m, s=sdist)
    return sd


def attach_ramp(J, lateral, nb):
    """Length of the chain attachment ramp in bone units. A sideways twist by angle t of the chain root moves a
    vertex at lateral distance x from the chain axis by about t * x along the chain, so an edge across the ramp
    stretches by t * x * (weight slope); with the smoothstep slope 1.5 / ramp the ramp must be at least
    1.5 * 2 sin(t / 2) * x / MAX_EDGE_STEP long. One bone (half the chain for one bone) is the minimum; capped at
    ATT_MAX_SHARE of the chain (wider panels need several chains, the caller warns)."""
    bl = float(np.mean(np.linalg.norm(J[1:] - J[:-1], axis=1)))
    hw = float(np.percentile(lateral, 95)) if len(lateral) else 0.0
    need = 1.5 * 2.0 * np.sin(np.radians(ATT_TWIST_DEG) / 2.0) * hw / MAX_EDGE_STEP / max(bl, 1e-12)
    lo = min(1.0, nb * 0.5)
    hi = max(lo, ATT_MAX_SHARE * nb)
    return float(min(max(lo, need), hi)), need > hi


# ------------------------------------------------------------------------------------------ stages 4-6: write

def bone_names(plan, sd):
    if plan['kind'] == 'COMPACT':
        return ["JK_%s%s" % (plan['base'], sd['suffix'])]
    return ["JK_%s_%d%s" % (plan['base'], k, sd['suffix']) for k in range(plan['nb'])]


def _mode(mode):
    bpy.ops.object.mode_set(mode=mode)


def write_bones(context, plan):
    """Stage 4. Returns [(chain names per side)] with the final bone names (Blender may add .001)."""
    arm, ob = plan['arm'], plan['ob']
    vl = context.view_layer
    if ob.mode != 'OBJECT':
        _mode('OBJECT')
    vl.objects.active = arm
    arm.select_set(True)
    _mode('EDIT')
    created = []
    try:
        eb = arm.data.edit_bones
        coll = arm.data.collections.get(COLLECTION) or arm.data.collections.new(COLLECTION)
        show_collection(coll, plan['warn'])
        for sd in plan['sides']:
            names = []
            prev = None
            J = sd['J']
            for k, want in enumerate(bone_names(plan, sd)):
                b = eb.new(want)
                b.head = Vector(J[k])
                b.tail = Vector(J[k + 1])
                if prev is None:
                    b.parent = eb.get(sd['parent']) if sd['parent'] else None
                    b.use_connect = False
                else:
                    b.parent = prev
                    b.use_connect = True
                b.use_deform = True
                axis = J[k + 1] - J[k]
                b.align_roll(Vector(regions.roll_up(axis, plan['up_z'], plan['up_y'])))
                coll.assign(b)
                names.append(b.name)
                created.append(b.name)
                prev = b
            sd['names'] = names
    finally:
        _mode('OBJECT')
        plan['created_bones'] = created
    # stage 4b: identity and chain parameters (Object.jigglekit.chains)
    for sd in plan['sides']:
        cid = new_id()
        sd['chain_id'] = cid
        pbs = [arm.pose.bones[nm] for nm in sd['names']]
        # the chain entry first: when it cannot be stored, no bone is left pointing at a chain that is not there
        add_chain(arm, cid, plan['base'] + sd['suffix'], plan['preset'],
                  float(np.mean([pb.bone.length for pb in pbs])))
        for k, pb in enumerate(pbs):
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            jb = pb.jigglekit
            jb.chain_id = cid
            jb.bone_id = new_id()
            jb.index = k
            jb.rest_q = (1.0, 0.0, 0.0, 0.0)


def add_chain(arm, cid, name, preset, mean_len):
    """New entry in Object.jigglekit.chains. Raises CreateError when the armature cannot store it (a library
    override on Blender 5.x, or a collection that refuses the insertion)."""
    from . import runtime
    jk = arm.jigglekit
    if model.override_unsaved(arm):
        raise CreateError(model.OVERRIDE_REFUSE % arm.name)
    try:
        ch = jk.chains.add()
    except TypeError:
        raise CreateError(model.OVERRIDE_REFUSE % arm.name)
    ch.id = cid
    ch.name = name[:40]
    presets.apply_to_chain(ch, preset, props.applying, mean_bone_length=mean_len)
    jk.chain_index = len(jk.chains) - 1
    runtime.invalidate()
    return ch


def remove_chains(arm, chain_ids):
    from . import runtime
    jk = arm.jigglekit
    for i in reversed(range(len(jk.chains))):
        if jk.chains[i].id in chain_ids:
            jk.chains.remove(i)
    jk.chain_index = min(jk.chain_index, max(0, len(jk.chains) - 1))
    runtime.invalidate()


def write_weights(context, plan):
    """Stage 5: CARVE on the touched vertices only, vertex_group.add / remove (spec 5.7)."""
    ob, arm = plan['ob'], plan['arm']
    me = ob.data
    sides = plan['sides']
    touched = np.zeros(len(me.vertices), dtype=bool)
    for sd in sides:
        touched |= sd['M'] > 0.0
    T = np.flatnonzero(touched)
    old = read_groups(me, T)
    names = list(plan['vg_names'])
    locked = list(plan['vg_locked'])
    used = sorted({g for gw in old.values() for g in gw})
    cols = {g: j for j, g in enumerate(used)}
    col_name = [names[g] for g in used]
    col_group = list(used)                       # vertex group index per column (-1 = to be created)
    col_locked = [locked[g] for g in used]
    new_cols_side = []
    for sd in sides:
        nc = []
        for nm in sd['names']:
            if nm in col_name:                   # group already has weights here (Weights for Selection again)
                nc.append(col_name.index(nm))
                continue
            col_name.append(nm)
            col_group.append(names.index(nm) if nm in names else -1)
            col_locked.append(False)
            nc.append(len(col_name) - 1)
        new_cols_side.append(nc)
    parent_cols = []
    for sd in sides:
        p = sd['parent']
        if p in col_name:
            parent_cols.append(col_name.index(p))
        else:
            col_name.append(p)
            col_group.append(names.index(p) if p in names else -1)
            col_locked.append(False)
            parent_cols.append(len(col_name) - 1)
    W = np.zeros((len(T), len(col_name)))
    for r, v in enumerate(T):
        for g, w in old[int(v)].items():
            W[r, cols[g]] = w
    W0 = W.copy()
    deform_bones = set(plan['deform_bones'])
    new_all = [j for nc in new_cols_side for j in nc]
    deform = np.array([col_name[j] in deform_bones and not col_locked[j] and j not in new_all
                       for j in range(len(col_name))], dtype=bool)
    Mc, Sc, side = weights.combine_sides([sd['M'][T] for sd in sides], [sd['shares'][T] for sd in sides])
    weights.carve(W, deform, Mc, Sc, new_all, np.asarray(parent_cols, dtype=np.int64)[side])
    if plan['max_influences'] > 0:
        # only deform-bone groups count and may be cut (the new bones included); pin, mask and other non-bone
        # groups and locked groups are never touched (review finding 5)
        free = np.array([(col_name[j] in deform_bones or j in new_all) and not col_locked[j]
                         for j in range(len(col_name))], dtype=bool)
        weights.limit_influences(W, free, plan['max_influences'])
    plan['skipped_locked'] = int(sum(1 for j in range(len(used)) if col_locked[j]))
    # write: create groups, then per column only the changed entries
    created_groups = []
    for j, nm in enumerate(col_name):
        if col_group[j] < 0:
            vg = ob.vertex_groups.get(nm)
            if vg is None:
                vg = ob.vertex_groups.new(name=nm)
                created_groups.append(nm)
            col_group[j] = vg.index
    plan['created_groups'] = created_groups
    plan['weights_backup'] = (T, W0, list(col_group))
    vgs = ob.vertex_groups
    changed = np.abs(W - W0) > 0.0
    for j in range(len(col_name)):
        rows = np.flatnonzero(changed[:, j])
        if len(rows) == 0:
            continue
        vg = vgs[col_group[j]]
        for r in rows:
            v = int(T[r])
            w = float(W[r, j])
            if w > 0.0:
                vg.add([v], w, 'REPLACE')
            elif W0[r, j] > 0.0:
                vg.remove([v])
    plan['touched'] = int(len(T))


def rollback(context, plan):
    arm, ob = plan['arm'], plan['ob']
    try:
        if context.object is not None and context.object.mode != 'OBJECT':
            _mode('OBJECT')
        bk = plan.get('weights_backup')
        if bk is not None:
            T, W0, col_group = bk
            for j, g in enumerate(col_group):
                if g < 0 or g >= len(ob.vertex_groups):
                    continue
                vg = ob.vertex_groups[g]
                for r, v in enumerate(T):
                    if W0[r, j] > 0.0:
                        vg.add([int(v)], float(W0[r, j]), 'REPLACE')
        for nm in plan.get('created_groups', []):
            vg = ob.vertex_groups.get(nm)
            if vg is not None:
                ob.vertex_groups.remove(vg)
        if plan.get('created_bones'):
            context.view_layer.objects.active = arm
            _mode('EDIT')
            eb = arm.data.edit_bones
            for nm in plan['created_bones']:
                b = eb.get(nm)
                if b is not None:
                    eb.remove(b)
            _mode('OBJECT')
            remove_chains(arm, {sd.get('chain_id') for sd in plan['sides']})
        context.view_layer.objects.active = ob
    except Exception as e:  # noqa: BLE001
        print("JiggleKit: rollback failed:", repr(e))


def execute_plan(context, plan):
    t0 = time.perf_counter()
    write_bones(context, plan)
    t1 = time.perf_counter()
    plan['touched'] = 0
    plan['skipped_locked'] = 0
    if plan['skinned'] and plan['weight_mode'] == 'CARVE':
        context.view_layer.objects.active = plan['ob']
        write_weights(context, plan)
    t2 = time.perf_counter()
    # stage 6: finish (Object Mode, mesh active, armature selected), start the live simulation
    vl = context.view_layer
    vl.objects.active = plan['ob']
    plan['ob'].select_set(True)
    plan['arm'].select_set(True)
    rt = model.rebuild(plan['arm'])
    plan['timings'].update(bones=t1 - t0, write_weights=t2 - t1)
    return rt


class Settings:
    """Create settings: an operator property set by the caller wins, else the Create panel (Scene.jigglekit_ui).
    The panel stores mirror as a bool and segments >= 1; when the caller picks another region than the panel
    shows, mirror and segments follow that region's preset."""

    def __init__(self, op, ui):
        def given(k):
            return op.properties.is_property_set(k)
        self.region = op.region if given('region') else ui.region
        follow_ui = self.region == ui.region
        self.armature = op.armature if given('armature') else (ui.armature.name if ui.armature else "")
        self.name = op.name if given('name') else ui.name
        if given('mirror'):
            self.mirror = op.mirror
        else:
            self.mirror = ('ON' if ui.mirror_x else 'OFF') if follow_ui else 'PRESET'
        if given('segments'):
            self.segments = op.segments
        else:
            self.segments = ui.segments if follow_ui else 0
        for k in ('weight_mode', 'falloff', 'core', 'weight_max', 'max_influences', 'parent_mode', 'limit_ratio'):
            setattr(self, k, getattr(op, k) if given(k) else getattr(ui, k))


SETTINGS = ('region', 'armature', 'name', 'mirror', 'segments', 'weight_mode', 'falloff', 'core', 'weight_max',
            'max_influences', 'parent_mode', 'limit_ratio')


class JIGGLEKIT_OT_create_from_selection(bpy.types.Operator):
    """Create jiggle bones, weights and physics from the selected vertices"""
    bl_idname = "jigglekit.create_from_selection"
    bl_label = "Create Jiggle from Selection"
    bl_options = {'REGISTER', 'UNDO'}

    region: EnumProperty(name="Region", default='CHEST',
                         items=[(k, presets.UI_LABELS[k], "") for k in presets.NAMES],
                         description="Region type. Sets bone count, mirroring and the physics preset",
                         options={'SKIP_SAVE'})
    armature: StringProperty(options={'SKIP_SAVE'}, name="Armature", default="",
                             description="Armature to add bones to. Empty = the armature of the mesh's Armature "
                                         "modifier")
    name: StringProperty(options={'SKIP_SAVE'}, name="Name", default="", maxlen=40,
                         description="Base name of the new bones. Empty = region label")
    mirror: EnumProperty(options={'SKIP_SAVE'}, name="Mirror X", default='PRESET',
                         items=[('PRESET', "Preset", ""), ('ON', "On", ""), ('OFF', "Off", "")],
                         description="Also create the mirrored side (armature X axis)")
    segments: IntProperty(options={'SKIP_SAVE'}, name="Segments", default=0, min=0, max=12,
                          description="Number of bones in a chain region. 0 = preset. Compact regions always get "
                                      "one bone")
    weight_mode: EnumProperty(options={'SKIP_SAVE'}, name="Weights", default='CARVE',
                              items=[('CARVE', "Carve", "Take weight from existing groups and give it to the new "
                                                        "bones; totals stay the same"),
                                     ('NONE', "None", "Create bones only")])
    falloff: FloatProperty(options={'SKIP_SAVE'}, name="Falloff", default=0.6, min=0.0, max=2.0,
                           description="How far weights spread outside the selection, as a fraction of the "
                                       "region radius")
    core: FloatProperty(options={'SKIP_SAVE'}, name="Core", default=0.4, min=0.05, max=1.0,
                        description="Inner ramp, as a fraction of the region radius")
    weight_max: FloatProperty(options={'SKIP_SAVE'}, name="Max Weight", default=1.0, min=0.1, max=1.0,
                              description="Highest weight the new bones can get")
    max_influences: IntProperty(options={'SKIP_SAVE'}, name="Max Influences", default=0, min=0, max=8,
                                description="Keep at most this many groups per vertex. 0 = do not limit")
    parent_mode: EnumProperty(options={'SKIP_SAVE'}, name="Parent", default='AUTO',
                              items=[('AUTO', "Auto", "Bone with the most weight under the selection"),
                                     ('ACTIVE_BONE', "Active Bone", "Active bone of the armature")])
    limit_ratio: FloatProperty(options={'SKIP_SAVE'}, name="Limit", default=0.6, min=0.1, max=0.95,
                               description="Refuse selections that cover more than this share of the mesh "
                                           "(a whole separate hair or tail mesh is accepted for chain regions)")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        if ob is None or ob.type != 'MESH' or ob.mode != 'EDIT':
            cls.poll_message_set("Enter Edit Mode on the body mesh and select the area")
            return False
        return True

    def execute(self, context):
        res = dict(ok=False, op="create_from_selection", error="", messages=[], counts={}, timings={})
        ob = context.active_object
        try:
            plan = plan_create(ob, Settings(self, context.scene.jigglekit_ui))
        except CreateError as e:
            res['error'] = str(e)
            write_result(context.scene, res)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        try:
            execute_plan(context, plan)
        except Exception as e:  # noqa: BLE001  nothing half-created stays (spec 5.7 stage 6)
            import traceback
            traceback.print_exc()
            rollback(context, plan)
            res['error'] = "Create failed: %r" % e
            write_result(context.scene, res)
            self.report({'ERROR'}, res['error'])
            return {'CANCELLED'}
        bones = [n for sd in plan['sides'] for n in sd['names']]
        if plan.get('skipped_locked'):
            plan['warn'].append("%d locked vertex groups were skipped." % plan['skipped_locked'])
        res.update(ok=True, messages=plan['warn'],
                   counts=dict(bones=len(bones), chains=len(plan['sides']), groups=len(plan.get('created_groups', [])),
                               vertices=plan['touched']),
                   bones=bones, chain_ids=[sd['chain_id'] for sd in plan['sides']],
                   parents=[sd['parent'] for sd in plan['sides']], parent_how=[sd['how'] for sd in plan['sides']],
                   joints=[np.round(sd['J'], 4).tolist() for sd in plan['sides']],
                   band=[np.round(sd['band'], 4).tolist() for sd in plan['sides']],
                   timings={k: round(v * 1000.0, 2) for k, v in plan['timings'].items()})
        write_result(context.scene, res)
        for w in plan['warn']:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, "JiggleKit: %d bones, %d chains, %d vertices weighted"
                    % (len(bones), len(plan['sides']), plan['touched']))
        return {'FINISHED'}


# ------------------------------------------------------------------------------------- Weights for Selection

def chain_armature(ob, ui):
    """Armature for a mesh: panel field, else the first Armature modifier."""
    arm, skinned = find_armature(ob, ui.armature.name if ui.armature else "")
    return arm, skinned


def active_chain(arm):
    jk = arm.jigglekit
    if not len(jk.chains):
        return None
    return jk.chains[min(jk.chain_index, len(jk.chains) - 1)]


def plan_paint(ob, ui):
    """Stages 1-3 of spec 5.7 for an existing chain on another mesh (clothing). Writes nothing."""
    t0 = time.perf_counter()
    warn = []
    arm, skinned = chain_armature(ob, ui)
    if arm is None:
        raise CreateError("No armature found. Add an Armature modifier or pick an armature in the panel.")
    ch = active_chain(arm)
    if ch is None:
        raise CreateError("The armature has no JiggleKit chains. Create one first.")
    if not skinned:
        raise CreateError("The chain bones are not in this mesh's armature.")
    bones = sorted(((pb.jigglekit.index, pb.name) for pb in arm.pose.bones if pb.jigglekit.chain_id == ch.id))
    names = [n for _i, n in bones]
    if not names:
        raise CreateError("The chain bones are not in this mesh's armature.")
    warn.extend(base_mesh_warnings(ob))
    co, nr, sel, edges = read_mesh(ob)
    nv = len(co)
    nsel = int(sel.sum())
    if nsel < 4:
        raise CreateError("Select at least 4 vertices.")
    db = arm.data.bones
    if nsel > ui.limit_ratio * nv:
        # most of the mesh: fine for a whole separate part (a hair mesh that follows the chain), not for a body
        sums = group_sums(ob.data, [vg.name for vg in ob.vertex_groups], {b.name for b in db}, np.flatnonzero(sel))
        segs = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in db}
        if regions.weight_line(sums, {b.name: b.parent.name if b.parent else None for b in db}, segs)[0] \
                < regions.LINE_SHARE:
            raise CreateError(whole_part_error(ui.limit_ratio, sums, "chain", active_hint=False))
    Mx = _np(arm.matrix_world.inverted()) @ _np(ob.matrix_world)
    A = co @ Mx[:3, :3].T + Mx[:3, 3]
    J = np.array([np.array(db[names[0]].head_local)] + [np.array(db[n].tail_local) for n in names])
    root = db[names[0]]
    parent = root.parent.name if root.parent is not None else None
    if parent is None:
        warn.append("The chain root has no parent bone; vertices without weights stay unweighted outside the area.")
    nb = len(names)
    kind = 'CHAIN' if nb > 1 else 'COMPACT'
    sd = dict(sel=sel, J=J, R=regions.bbox_half_diag(A[sel]), parent=parent, how='CHAIN', x=0.0, names=names,
              suffix="")
    G = weights.graph(nv, edges, A)
    side_weights(G, edges, A, sd, kind, nb, ui.core, ui.falloff, ui.weight_max, warn)
    return dict(ob=ob, arm=arm, skinned=True, kind=kind, nb=nb, sides=[sd], warn=warn, chain=ch.name,
                vg_names=[vg.name for vg in ob.vertex_groups], vg_locked=[vg.lock_weight for vg in ob.vertex_groups],
                deform_bones={b.name for b in db if b.use_deform}, weight_mode='CARVE',
                max_influences=ui.max_influences, timings=dict(read_geo_weights=time.perf_counter() - t0))


class JIGGLEKIT_OT_paint_weights(bpy.types.Operator):
    """Give the selected vertices of this mesh weights for the active chain, with the same smooth falloff as
    Create (for clothing or a second mesh)"""
    bl_idname = "jigglekit.paint_weights"
    bl_label = "Weights for Selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        if ob is None or ob.type != 'MESH' or ob.mode != 'EDIT':
            cls.poll_message_set("Enter Edit Mode on the mesh and select the area")
            return False
        arm, _sk = chain_armature(ob, context.scene.jigglekit_ui)
        if arm is None or not len(arm.jigglekit.chains):
            cls.poll_message_set("Needs an armature with a JiggleKit chain (pick the chain in Chains)")
            return False
        return True

    def execute(self, context):
        res = dict(ok=False, op="paint_weights", error="", messages=[], counts={}, timings={})
        ob = context.active_object
        try:
            plan = plan_paint(ob, context.scene.jigglekit_ui)
        except CreateError as e:
            res['error'] = str(e)
            write_result(context.scene, res)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
            write_weights(context, plan)
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            plan['created_bones'] = []
            rollback(context, plan)
            res['error'] = "Weights failed: %r" % e
            write_result(context.scene, res)
            self.report({'ERROR'}, res['error'])
            return {'CANCELLED'}
        if plan.get('skipped_locked'):
            plan['warn'].append("%d locked vertex groups were skipped." % plan['skipped_locked'])
        sd = plan['sides'][0]
        res.update(ok=True, messages=plan['warn'], bones=sd['names'], chain=plan['chain'],
                   counts=dict(vertices=plan['touched'], groups=len(plan.get('created_groups', []))),
                   band=np.round(sd['band'], 4).tolist(),
                   timings={k: round(v * 1000.0, 2) for k, v in plan['timings'].items()})
        write_result(context.scene, res)
        for w in plan['warn']:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, "JiggleKit: %d vertices weighted for %s" % (plan['touched'], plan['chain']))
        return {'FINISHED'}


# ------------------------------------------------------------------------------ Add Chains from Selected Bones

def bone_paths(sel_pbs):
    """Split selected pose bones into parent -> child paths. A bone with several selected children ends its path;
    each child starts a new one."""
    S = {pb.name for pb in sel_pbs}

    def kids(pb):
        return [c for c in pb.children if c.name in S]

    roots = [pb for pb in sel_pbs if pb.parent is None or pb.parent.name not in S or len(kids(pb.parent)) > 1]
    paths = []
    for r in sorted(roots, key=lambda b: b.name):
        path = [r]
        while True:
            k = kids(path[-1])
            if len(k) != 1:
                break
            path.append(k[0])
        paths.append(path)
    return paths


ROT_REF = re.compile(r'pose\.bones\["((?:[^"\\]|\\.)*)"\]\.rotation_(?:euler|axis_angle)')
EULER_HINT = 'Use "Convert to Quaternion and Add Chains" in the Result box.'
SCALE_HINT = 'Use "Set Inherit Scale to Full and Add Chains" in the Result box.'


def rotation_users(arm):
    """Names of the bones whose Euler / axis-angle rotation is animated, driven or read by a driver of this
    armature: switching such a bone to Quaternion would drop that animation, so it is not done for the user."""
    out = set()
    ad = arm.animation_data
    if ad is None:
        return out
    paths = []
    for act in [ad.action] + [st.action for tr in ad.nla_tracks for st in tr.strips]:
        paths.extend(fc.data_path for fc in bake.action_fcurves(act))
    for d in ad.drivers:
        paths.append(d.data_path)
        for v in d.driver.variables:
            paths.extend(t.data_path for t in v.targets if t.id == arm and t.data_path)
    for p in paths:
        m = ROT_REF.search(p)
        if m:
            out.add(re.sub(r'\\(.)', r'\1', m.group(1)))
    return out


def to_quaternion(pb):
    """Switch a pose bone to Quaternion rotation and keep its pose (setting rotation_mode from Python does not
    convert the values, unlike the menu in the N panel)."""
    if pb.rotation_mode == 'AXIS_ANGLE':
        a = pb.rotation_axis_angle
        ax = Vector(a[1:4])
        q = Quaternion(ax.normalized(), a[0]) if ax.length > 1e-12 else Quaternion()
    else:
        q = pb.rotation_euler.to_quaternion()
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = q


def add_chains(arm, preset, sel_names, convert=False, full_scale=False, scale=None):
    """Chains on existing bones (spec 4.3). Returns (chain names, bone count, messages, euler). Weights are not
    touched. convert: bones in an Euler / axis-angle mode are switched to Quaternion (pose kept) when nothing
    animates that rotation; euler = the bones that stopped a chain because of their rotation mode.
    Inherit Scale (model.inherit_scale_issue): a mode other than Full is taken as it is while it gives the same
    pose as Full; when it does not, the chain is refused (the bone's name is appended to the list `scale`), or,
    with full_scale, the bones of that chain are set to Full."""
    msgs = []
    euler = []
    scale = [] if scale is None else scale
    busy = None
    pbs = arm.pose.bones
    sel = []
    for n in sel_names:
        pb = pbs[n]
        if pb.jigglekit.chain_id:
            msgs.append("Bone %s already belongs to a chain; skipped." % n)
        else:
            sel.append(pb)
    paths = bone_paths(sel)
    if len(paths) > 1:
        msgs.append("Selected bones do not form a parent-child path; created %d chains." % len(paths))
    made, n_bones = [], 0
    coll = arm.data.collections.get(COLLECTION)
    if coll is None and not model.bones_locked(arm):        # linked armature data: no new bone collection
        coll = arm.data.collections.new(COLLECTION)
    shown = False
    for path in paths:
        bad = None
        refit = False
        for pb in path:
            issue = model.inherit_issue(pb)
            if issue and issue[0] == 'ROTATION':
                bad = "%s %s" % (issue[1], model.ROT_FIX)
                break
            if issue:                                   # an Inherit Scale mode that differs from Full here
                if not full_scale:
                    bad = "%s %s" % (issue[1], SCALE_HINT)
                    scale.append(pb.name)
                    break
                if model.bones_locked(arm):
                    bad = "%s The armature data is linked: set Inherit Scale to Full in the library file." % issue[1]
                    break
                refit = True
            if pb.rotation_mode != 'QUATERNION':
                if not convert:
                    bad = "Bone %s uses Euler/axis-angle rotation; the chain was skipped. %s" % (pb.name, EULER_HINT)
                    euler.append(pb.name)
                    break
                if busy is None:
                    busy = rotation_users(arm)
                if pb.name in busy:
                    bad = ("Bone %s has keyframes or drivers on its Euler/axis-angle rotation; the chain was "
                           "skipped. Switching it to Quaternion would drop that animation." % pb.name)
                    break
        if bad:
            msgs.append(bad)
            continue
        if refit:            # every non-Full bone of the path: a bone set to Full changes its children's parent
            fixed = [pb for pb in path if pb.bone.inherit_scale != 'FULL']
            for pb in fixed:
                pb.bone.inherit_scale = 'FULL'
            msgs.append("Set Inherit Scale to Full (the pose under a scaled parent changes): %s."
                        % ", ".join(pb.name for pb in fixed))
        turned = [pb for pb in path if pb.rotation_mode != 'QUATERNION']
        for pb in turned:
            to_quaternion(pb)
        if turned:
            msgs.append("Switched to Quaternion rotation (pose kept): %s." % ", ".join(pb.name for pb in turned))
        for pb in path:
            note = model.constraint_note(pb)
            if note:
                msgs.append(note)
        for n, owner_name in model.ik_over(arm, [pb.name for pb in path]):
            msgs.append("Bone %s is in the chain of the IK constraint on %s: IK overrides the jiggle. Jiggle "
                        "bones should not be part of an IK chain." % (n, owner_name))
        cid = new_id()
        name = path[0].name if len(path) == 1 else "%s (%d)" % (path[0].name, len(path))
        try:                 # the chain entry first: a refusal leaves the bones untouched (no orphan bones)
            add_chain(arm, cid, name, preset, float(np.mean([pb.bone.length for pb in path])))
        except CreateError as e:
            if str(e) not in msgs:
                msgs.append(str(e))
            break
        for k, pb in enumerate(path):
            jb = pb.jigglekit
            jb.chain_id = cid
            jb.index = k
            jb.bone_id = new_id()
            jb.rest_q = tuple(pb.rotation_quaternion)
            if coll is not None:
                coll.assign(pb)
        if not shown:
            show_collection(coll, msgs)
            shown = True
        made.append(name)
        n_bones += len(path)
    return made, n_bones, msgs, euler


def hidden_selected(arm):
    """Number of selected pose bones (meant for the case when context.selected_pose_bones is empty: these are in
    hidden bone collections or hidden themselves, so Blender does not hand them to operators)."""
    n = 0
    for pb in arm.pose.bones:
        n += bool(pb.select if hasattr(pb, "select") else pb.bone.select)      # 5.x: selection on the pose bone
    return n


class JIGGLEKIT_OT_add_chains_from_bones(bpy.types.Operator):
    """Make jiggle chains from the selected bones of an existing rig (hair, tail). Weights are not changed"""
    bl_idname = "jigglekit.add_chains_from_bones"
    bl_label = "Add Chains from Selected Bones"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="Preset", items=[(k, presets.UI_LABELS[k], "") for k in presets.NAMES],
                         default='HAIR_STRAND', options={'SKIP_SAVE'},
                         description="Physics preset of the new chains. Default: the Region of the Create panel")
    convert_rotation: BoolProperty(name="Convert to Quaternion", default=False, options={'SKIP_SAVE'},
                                   description="Switch selected bones that use Euler or axis-angle rotation (FK "
                                               "controls of Rigify, CloudRig and similar rigs) to Quaternion and "
                                               "keep their pose. Bones with keyframes or drivers on that rotation "
                                               "are left alone")
    fix_inherit_scale: BoolProperty(name="Set Inherit Scale to Full", default=False, options={'SKIP_SAVE'},
                                    description="Set Inherit Scale to Full on selected bones whose mode (Aligned, "
                                                "None and others) gives a different pose than Full because their "
                                                "parent is scaled. The pose of these bones changes. Bones under "
                                                "an unscaled parent are simulated as they are")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        if ob is None or ob.type != 'ARMATURE' or ob.mode != 'POSE':
            cls.poll_message_set("Select bones in Pose Mode")
            return False
        if not context.selected_pose_bones:
            n = hidden_selected(ob)
            cls.poll_message_set("The %d selected bones are hidden: show their bone collection (Armature properties "
                                 "> Bone Collections) and select them again" % n if n
                                 else "Select bones in Pose Mode (no visible bone is selected)")
            return False
        return True

    def execute(self, context):
        ob = context.active_object
        preset = self.preset if self.properties.is_property_set('preset') else context.scene.jigglekit_ui.region
        names = [pb.name for pb in context.selected_pose_bones if pb.id_data == ob]
        if EDITION == 'DEMO':
            have = model.demo_bones_in_file()
            if have + len(names) > DEMO_BONE_LIMIT:
                write_result(context.scene, dict(ok=False, op="add_chains_from_bones", messages=[], counts={},
                                                 error="Demo limit: %d jiggle bones. Get the full version."
                                                 % DEMO_BONE_LIMIT))
                self.report({'ERROR'}, "Demo limit: %d jiggle bones. Get the full version." % DEMO_BONE_LIMIT)
                return {'CANCELLED'}
        scale = []
        made, n_bones, msgs, euler = add_chains(ob, preset, names, self.convert_rotation, self.fix_inherit_scale,
                                                scale)
        model.rebuild(ob)
        refused = model.OVERRIDE_REFUSE % ob.name
        res = dict(ok=bool(made), op="add_chains_from_bones",
                   error="" if made else (refused if refused in msgs else "No chain was created."),
                   messages=msgs, counts=dict(chains=len(made), bones=n_bones), chains=made, euler=euler,
                   inherit_scale=scale)
        write_result(context.scene, res)
        for m in msgs:
            self.report({'WARNING'}, m)
        if not made:
            self.report({'ERROR'}, res['error'])
            return {'CANCELLED'}
        self.report({'INFO'}, "JiggleKit: %d chains, %d bones" % (len(made), n_bones))
        return {'FINISHED'}


CLASSES = (JIGGLEKIT_OT_create_from_selection, JIGGLEKIT_OT_paint_weights, JIGGLEKIT_OT_add_chains_from_bones)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

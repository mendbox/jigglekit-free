# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuild of the live runtime from stored data (spec 5.8, 6.1): PoseBone.jigglekit (chain_id, index, rest_q,
bone_id) plus Object.jigglekit (chains, colliders, armature settings). Runs duplicate-ID repair first.

A chain is cut at the first gap in the indices or parent link; bones of chains without an entry in
Object.jigglekit.chains are orphans (not simulated, note in Result)."""
from . import ids, runtime
from .const import DEMO_BONE_LIMIT, EDITION

SIM_KEYS = ('stiffness', 'damping', 'inertia', 'gravity', 'angle_limit', 'tip_ratio', 'tip_damping', 'buckle',
            'radius', 'collide', 'influence', 'rest_contact')


# Library overrides (review finding 3, measured 2026-11-23). Blender 4.2 / 4.5 store every JiggleKit property of an
# override one by one (and inserted chains / colliders: USE_INSERTION), so everything survives save and reload.
# Blender 5.2 records one whole-struct REPLACE for Object.jigglekit and PoseBone.jigglekit, which restores
# nothing on reload: chains or colliders added to an override, and any setting changed on it, are lost with the
# next reload (explicit override operations are purged on save). So on 5.x adding is refused and a note says
# that changes are not saved; rigs whose chains were made in the library file play normally.
OVERRIDE_REFUSE = ("%s is a library override. Blender 5 does not keep JiggleKit chains or colliders added to an "
                   "override: add them in the library file.")
OVERRIDE_NOTE = ("Library override: Blender 5 does not save JiggleKit changes made here. Edit the rig in its "
                 "library file.")
LINKED_BONES = ("The armature %s is linked from a library: new bones cannot be added here. Create the jiggle in "
                "the library file, or use Chains from Selected Bones on existing bones.")


def override_unsaved(ob):
    """True when JiggleKit data written to this object is lost on reload (a library override on Blender 5.x)."""
    import bpy
    return ob.override_library is not None and bpy.app.version >= (5, 0, 0)


def bones_locked(arm):
    """True when bones cannot be added to the armature (linked or override data: no Edit Mode)."""
    d = arm.data
    return arm.library is not None or d.library is not None or d.override_library is not None


def _chain_bones(ob):
    groups = {}
    for pb in ob.pose.bones:
        jb = pb.jigglekit
        if jb.chain_id:
            groups.setdefault(jb.chain_id, []).append((jb.index, pb.name))
    return groups


# Section 12 item 10 (measured, docs/PROTOTYPE-NOTES.md days 11-13): a constraint on a simulated bone acts after
# the pose the add-on writes, so these replace the jiggled rotation (the jiggle is lost or the bone is moved off
# the simulation); Limit Rotation clamps it; the rest may offset it. Constraints on the chain's parent bones are
# fine (they are read with the evaluated pose).
OVERRIDING = {'COPY_ROTATION', 'COPY_TRANSFORMS', 'DAMPED_TRACK', 'TRACK_TO', 'LOCKED_TRACK', 'STRETCH_TO',
              'CHILD_OF', 'TRANSFORM', 'ACTION', 'ARMATURE', 'IK', 'SPLINE_IK', 'CLAMP_TO', 'FOLLOW_PATH',
              'PIVOT', 'SHRINKWRAP', 'FLOOR', 'OBJECT_SOLVER', 'FOLLOW_TRACK', 'GEOMETRY_ATTRIBUTE',
              'TRANSFORM_CACHE', 'CAMERA_SOLVER'}


def _active(c):
    return not getattr(c, 'mute', False) and getattr(c, 'enabled', True) and c.influence > 0.0


def _label(c):
    return c.bl_rna.properties['type'].enum_items[c.type].name if c.type in \
        c.bl_rna.properties['type'].enum_items else c.type.title()


def constraint_note(pb):
    """Spec 7 / item 10: the warning for constraints on a simulated bone (None when there are none active)."""
    cons = [c for c in pb.constraints if _active(c)]
    if not cons:
        return None
    over = [c for c in cons if c.type in OVERRIDING]
    if over:
        return ("Bone %s: its %s constraint overrides the jiggle. Jiggle the bone it follows instead, or mute the "
                "constraint." % (pb.name, _label(over[0])))
    if any(c.type == 'LIMIT_ROTATION' for c in cons):
        return "Bone %s: its Limit Rotation constraint clamps the jiggle." % pb.name
    return "Bone %s has constraints; the result may be offset." % pb.name


def ik_over(ob, names):
    """Simulated bones that are inside the IK chain of another bone's IK constraint: [(bone, IK owner)]."""
    out = []
    want = set(names)
    for pb in ob.pose.bones:
        for c in pb.constraints:
            if c.type not in ('IK', 'SPLINE_IK') or not _active(c) or pb.name in want:
                continue
            n = c.chain_count
            p, k = pb.parent if c.type == 'IK' and not getattr(c, 'use_tail', True) else pb, 0
            while p is not None and (n == 0 or k < n):
                if p.name in want and p.name != pb.name:
                    out.append((p.name, pb.name))
                p, k = p.parent, k + 1
    return out


def nonuniform_above(ob, pb):
    """Spec 7: non-uniform scale of the object or of the parent bones of a chain root (current pose)."""
    import numpy as np
    M = np.array(ob.matrix_world, dtype=np.float64)[:3, :3]
    if pb.parent is not None:
        M = M @ np.array(pb.parent.matrix, dtype=np.float64)[:3, :3]
    sc = np.linalg.norm(M, axis=0)
    return bool(sc.max() > 0 and np.ptp(sc) > 1e-3 * sc.max())


# Inherit Scale of chain bones (showcase issues E and F, 2026-11-24). The solver computes a bone's frame as parent
# frame @ rest offset @ pose (sim.py), which is Blender's Inherit Scale "Full". The other modes give exactly the
# same matrix while the parent's pose matrix (armature space, so the object's own scale does not count) has the
# properties below, so such bones are simulated as they are; otherwise the written rotation would not put the bone
# where the solver thinks it is, and the chain is refused with the reason.
#   Fix Shear, Aligned, Average: the parent has no shear and one scale on all three axes (any size).
#   None, None (Legacy): the parent has no shear and scale 1.
# (Blender: BKE_bone_parent_transform_calc_from_matrices.) Checked on the current pose, like nonuniform_above().
SCALE_LABELS = {'FULL': "Full", 'FIX_SHEAR': "Fix Shear", 'ALIGNED': "Aligned", 'AVERAGE': "Average",
                'NONE': "None", 'NONE_LEGACY': "None (Legacy)"}
SCALE_TOL = 1e-3
ROT_FIX = "Turn on Inherit Rotation (Bone properties > Relations)."
SCALE_FIX = "Set Inherit Scale to Full (Bone properties > Relations) or clear the parent's scale."


def inherit_scale_issue(pb):
    """None when the bone's Inherit Scale mode gives the same pose as Full under its parent's current pose;
    else the reason as text ("Inherit Scale is Aligned and its parent Head is scaled non-uniformly")."""
    import numpy as np
    b = pb.bone
    mode = b.inherit_scale
    if mode == 'FULL' or pb.parent is None:
        return None
    M = np.array(pb.parent.matrix, dtype=np.float64)[:3, :3]
    sc = np.linalg.norm(M, axis=0)
    label = SCALE_LABELS.get(mode, mode.title())
    if sc.min() <= 0.0 or np.linalg.det(M) <= 0.0:
        return "Inherit Scale is %s and its parent %s has a zero or negative scale" % (label, pb.parent.name)
    G = (M / sc).T @ (M / sc)
    if np.ptp(sc) > SCALE_TOL * sc.max() or np.abs(G - np.eye(3)).max() > SCALE_TOL:
        return "Inherit Scale is %s and its parent %s is scaled non-uniformly" % (label, pb.parent.name)
    if mode in ('NONE', 'NONE_LEGACY') and abs(sc.mean() - 1.0) > SCALE_TOL:
        return "Inherit Scale is %s and its parent %s is scaled" % (label, pb.parent.name)
    return None


def inherit_issue(pb):
    """Why a bone cannot be simulated because of its Relations settings: (kind, text) with kind 'ROTATION' or
    'SCALE', or None. One cause per message, the bone is named (issue F)."""
    if not pb.bone.use_inherit_rotation:
        return 'ROTATION', "Bone %s has Inherit Rotation off; the chain was skipped." % pb.name
    why = inherit_scale_issue(pb)
    if why:
        return 'SCALE', "Bone %s: %s; the chain was skipped." % (pb.name, why)
    return None


# Free edition (spec 9.3): at most DEMO_BONE_LIMIT simulated bones in the whole file. Armatures take the budget in
# name order, inside an armature the chains in their list order and the bones root first. {armature name: (kept,
# stored)} of the last collect() of a capped armature, for the main panel.
DEMO_NOTE = "Free version: %d of %d jiggle bones simulated (limit %d). Get the full version for more."
CAPPED = {}


def demo_bones_in_file():
    """Chain bones stored in every armature of the file (what the Free edition's create operators count)."""
    import bpy
    return sum(1 for o in bpy.data.objects if o.type == 'ARMATURE' and o.pose is not None
               for pb in o.pose.bones if pb.jigglekit.chain_id)


def _demo_budget(ob):
    import bpy
    used = 0
    for o in sorted((o for o in bpy.data.objects if o.type == 'ARMATURE' and o.name < ob.name
                     and len(o.jigglekit.chains)), key=lambda o: o.name):
        used += sum(len(c['bones']) for c in collect(o, cap=False)['chains'])
    return max(0, DEMO_BONE_LIMIT - used)


def _demo_cap(ob, chains, notes, idle):
    """Cut the simulated bones to the Free edition's budget (only the list is cut; nothing stored changes)."""
    budget = _demo_budget(ob)
    total = sum(len(c['bones']) for c in chains)
    if total <= budget:
        CAPPED.pop(ob.name, None)
        return chains
    order = {ch.id: i for i, ch in enumerate(ob.jigglekit.chains)}
    keep, left = {}, budget
    for c in sorted(chains, key=lambda c: order.get(c['chain_id'], len(order))):
        keep[c['chain_id']] = min(left, len(c['bones']))
        left -= keep[c['chain_id']]
    out = []
    for c in chains:
        k = keep[c['chain_id']]
        idle.extend(c['bones'][k:])
        if k:
            c['bones'] = c['bones'][:k]
            out.append(c)
    notes.insert(0, DEMO_NOTE % (budget, total, DEMO_BONE_LIMIT))
    CAPPED[ob.name] = (budget, total)
    return out


def contact_of(pb, by_bid, sim_names):
    """(body bone name, normal) of a chain bone that lies on the body in the rest pose (stored by
    create_ops.measure_contact), or None: nothing stored, the body bone is gone, or it is simulated itself."""
    jb = pb.jigglekit
    n = tuple(jb.contact_n)
    if not jb.contact_bone or not any(n):
        return None
    p = by_bid.get(jb.contact_bone)
    if p is None:
        return None
    name = p.name
    while p is not None:
        if p.name in sim_names:
            return None
        p = p.parent
    return name, n


def collect(ob, cap=True):
    """Chains (dicts for runtime.ArmRT), colliders, rest quaternions and notes. Reads only. cap=False: without the
    Free edition's bone limit (used to count what other armatures take)."""
    jk = ob.jigglekit
    pbs = ob.pose.bones
    params = {ch.id: ch for ch in jk.chains}
    chains, rest, notes, idle = [], [], [], []
    broken = 0
    by_bid = {pb.jigglekit.bone_id: pb for pb in pbs if pb.jigglekit.bone_id}
    sim_all = {pb.name for pb in pbs if pb.jigglekit.chain_id}
    for cid, lst in _chain_bones(ob).items():
        ch = params.get(cid)
        lst.sort()
        if ch is None:
            notes.append("Orphan bones %s: their chain is gone (Repair Chains unlinks them)."
                         % ", ".join(n for _i, n in lst[:3]))
            idle.extend(n for _i, n in lst)
            continue
        names = []
        for k, (idx, name) in enumerate(lst):
            pb = pbs[name]
            if idx != k or (k > 0 and (pb.parent is None or pb.parent.name != names[-1])):
                notes.append("Chain %s broken at bone index %d." % (ch.name, k))
                broken += 1
                break
            names.append(name)
        idle.extend(n for _i, n in lst if n not in names)
        if not names:
            continue
        if not ch.enabled:
            idle.extend(names)
            continue
        skip = None
        for n in names:
            pb = pbs[n]
            bad = inherit_issue(pb)
            if bad:
                skip = "%s %s" % (bad[1], ROT_FIX if bad[0] == 'ROTATION' else SCALE_FIX)
                break
            if pb.rotation_mode != 'QUATERNION':
                skip = ("Bone %s uses Euler/axis-angle rotation; the chain was skipped. Set its rotation mode to "
                        "Quaternion." % n)
                break
            note = constraint_note(pb)
            if note:
                notes.append(note)
        if skip:
            notes.append(skip)
            idle.extend(names)
            continue
        if nonuniform_above(ob, pbs[names[0]]):
            notes.append("Non-uniform scale above bone %s: the result is approximate. Apply the scale." % names[0])
        for n, owner_name in ik_over(ob, names):
            notes.append("Bone %s is in the chain of the IK constraint on %s: IK overrides the jiggle. Jiggle "
                         "bones should not be part of an IK chain." % (n, owner_name))
        d = {k: getattr(ch, k) for k in SIM_KEYS}
        if ch.mesh_radius:        # radius measured around each bone (Create, Weights for Selection); 0 = not measured
            rr = [pbs[n].jigglekit.radius for n in names]
            if max(rr) > 0.0:
                d['radii'] = [r if r > 0.0 else ch.radius for r in rr]
        if ch.rest_contact and ch.collide:      # rest spheres: bones that lie on the body in the rest pose
            cc = [contact_of(pbs[n], by_bid, sim_all) for n in names]
            if any(cc):
                d['contacts'] = cc
        d['bones'] = names
        d['chain_id'] = cid
        chains.append(d)
    if EDITION == 'DEMO' and cap:
        chains = _demo_cap(ob, chains, notes, idle)
    rest = [tuple(pbs[n].jigglekit.rest_q) for c in chains for n in c['bones']]
    sim_names = {n for c in chains for n in c['bones']}
    colliders = []
    for ci, col in enumerate(jk.colliders):
        if not col.enabled:
            continue
        if col.bind == 'OBJECT':
            if col.object is None:
                notes.append("Collider %s has no object; it is off." % col.name)
                continue
            # the runtime reads the object from the pointer property (by list index) on every frame: the name
            # is not an identity (renames, same name in two libraries)
            colliders.append(dict(object=col.object.name, object_index=ci, radius=col.radius, shape='SPHERE',
                                  offset=tuple(col.offset), name=col.name))
            continue
        pb = by_bid.get(col.bone_id)
        if pb is None:
            notes.append("Collider %s has no bone; it is off." % col.name)
            continue
        p, bad = pb, False
        while p is not None:
            if p.name in sim_names:
                bad = True
                break
            p = p.parent
        if bad:
            notes.append("Collider %s: collider bone is simulated or is a child of a simulated bone; it is off."
                         % col.name)
            continue
        colliders.append(dict(bone=pb.name, radius=col.radius, shape=col.shape, t0=col.t0,
                              t1=col.t1 if col.shape == 'CAPSULE' else col.t0, offset=tuple(col.offset),
                              name=col.name))
    if override_unsaved(ob):
        notes.append(OVERRIDE_NOTE)
    n_sim = len(sim_names)
    if n_sim > 300:
        notes.append("Live simulation may be slow; bake recommended.")
    return dict(chains=chains, colliders=colliders, rest=rest, notes=notes, idle=idle, broken=broken,
                ok=len(chains))


def describe(ob):
    c = collect(ob)
    return dict(ok=c['ok'], broken=c['broken'], notes=c['notes'])


def rebuild(ob):
    """Build (or drop) the runtime of one armature from its stored data. Returns the ArmRT or None."""
    old = runtime.RT.get(runtime.key(ob))
    runtime.invalidate()
    jk = ob.jigglekit
    ids.migrate(ob)
    n_dup, new_chains = ids.repair_duplicates(ob)
    c = collect(ob)
    notes = list(c['notes'])
    if n_dup:
        notes.insert(0, "%d duplicated IDs fixed%s." % (
            n_dup, (", new chains: " + ", ".join(new_chains)) if new_chains else ""))
    runtime.restore_rest(ob, c['idle'])
    if not c['chains']:
        if old is not None and getattr(old, 'from_props', False):
            runtime.RT.pop(runtime.key(ob), None)
        return None
    rt = runtime.attach(ob, c['chains'], c['colliders'], rest_q=c['rest'], substeps=jk.substeps,
                        gravity_vector=tuple(jk.gravity_vector), time_scale=jk.time_scale,
                        max_catchup=jk.max_catchup)
    rt.notes = notes
    rt.from_props = True
    rt.ptr = ob.as_pointer()
    rt.built_version = jk.version
    rt.chain_ids = [ch['chain_id'] for ch in c['chains']]
    rt.use_cache = jk.use_cache
    rt.cache_max = jk.cache_max_frames
    if old is not None:
        rt.adopt(old)
    return rt

# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake to keyframes (spec 5.6).

record(): the live handler runs in recording mode (runtime.ArmRT.recording: no cache, no restart at the scene
start, the simulated quaternions of every frame go to rt.record) while frames start - warmup .. end are set one
by one. write_nla() / write_action() turn the recorded quaternions into LINEAR keys on rotation_quaternion
(4 F-curves per bone, grouped by bone name, points added in one batch). clear() removes what a bake made.

Action API (docs/PROTOTYPE-NOTES.md, section 12 item 4): 4.4+ (feature test 'slots' in Action RNA) uses layered
actions: slot, layer, keyframe strip, channelbag, F-curves; the NLA strip picks the slot up by itself. 4.2 uses
action.fcurves. 4.5 has both; the feature test picks the layered path there."""
import re

import bpy
import numpy as np

from . import ids, model, runtime
from .const import DEMO_BAKE_FRAMES, EDITION, FULL_URL

TRACK = "JiggleKit Bake"
TAG = "jigglekit_bake"             # custom property on the bake action: the bake_id
TAG_KEYS = "jigglekit_bake_keys"   # custom property on the user's action a Target: Action bake wrote into
ROT_PATH = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(rotation_quaternion|rotation_euler|rotation_axis_angle)$')
_LINEAR = None


class BakeError(Exception):
    pass


def layered():
    return 'slots' in bpy.types.Action.bl_rna.properties


def frame_range(scene, jk):
    if jk.bake_use_scene_range:
        return scene.frame_start, scene.frame_end
    return jk.bake_start, jk.bake_end


# Free edition: a bake covers at most DEMO_BAKE_FRAMES frames, the first ones of the range. The limit is applied
# here (free_range, record, _curves), not in the panel, so a script that calls the operator or these functions
# gets the same result. The live simulation is never limited.
FREE_BAKE_NOTE = ("Free version: bake up to %d frames (you asked for %d), so the first %d were baked. "
                  "Full version: unlimited, " + FULL_URL)


def free_range(start, end):
    """(last frame that is baked, number of frames asked for). Full edition: the range as it is. Free edition:
    the first DEMO_BAKE_FRAMES frames of a longer range."""
    asked = end - start + 1
    if EDITION == 'DEMO' and asked > DEMO_BAKE_FRAMES:
        return start + DEMO_BAKE_FRAMES - 1, asked
    return end, asked


def free_bake_note(asked):
    return FREE_BAKE_NOTE % (DEMO_BAKE_FRAMES, asked, DEMO_BAKE_FRAMES)


def key_frames(start, end, step):
    fr = list(range(start, end + 1, step))
    if fr and fr[-1] != end:
        fr.append(end)                      # the last frame of the range is always keyed
    return fr


def bone_path(name):
    return 'pose.bones["%s"].rotation_quaternion' % bpy.utils.escape_identifier(name)


def _unescape(s):
    return re.sub(r'\\(.)', r'\1', s)


def action_fcurves(act, slot=None):
    """F-curves of an action: all channelbags of the slot (layered, 4.4+) or action.fcurves (4.2)."""
    if act is None:
        return []
    if layered() and hasattr(act, "layers"):
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for cb in getattr(strip, "channelbags", ()):
                    if slot is None or cb.slot == slot:
                        out.extend(cb.fcurves)
        return out
    return list(act.fcurves)


def keyed_rotation_bones(act, slot, names):
    """Bones from names with rotation F-curves in the action (slot)."""
    want = set(names)
    hit = []
    for fc in action_fcurves(act, slot):
        m = ROT_PATH.match(fc.data_path)
        if m:
            n = _unescape(m.group(1))
            if n in want and n not in hit:
                hit.append(n)
    return hit


def chain_bone_names(ob):
    """Simulated bones of the armature (stored chains, enabled, in the rebuild order)."""
    return [n for c in model.collect(ob)['chains'] for n in c['bones']]


# ---------------------------------------------------------------------------------------------------- record

def record(scene, ob, start, end, warmup, progress=None, fail_at=None):
    """Run the simulation over start - warmup .. end through the frame handler. Returns (names, Q) with
    Q (end - start + 1, N, 4) float64 (sign-continuous, spec 5.2). The caller restores the frame. Not for an
    armature with an active bake (the baked keys would be simulated twice). Free edition: end is cut by
    free_range(), so Q has at most DEMO_BAKE_FRAMES frames."""
    end, _asked = free_range(start, end)
    if ob.jigglekit.use_baked:
        raise BakeError("Bake already exists. Clear it first.")
    runtime.RT.pop(runtime.key(ob), None)
    runtime.BAKING.add(runtime.key(ob))
    try:
        rt = model.rebuild(ob)
        if rt is None:
            raise BakeError("No chains to bake.")
        rt.recording = True
        rt.record = {}
        rt.error = ""
        f0 = start - warmup
        total = end - f0 + 1
        for k, f in enumerate(range(f0, end + 1)):
            if fail_at is not None and f == fail_at:
                raise BakeError("Test failure at frame %d." % f)      # bake_exception check
            scene.frame_set(f)
            if rt.error:
                raise BakeError("Simulation failed at frame %d: %s" % (f, rt.error))
            if runtime.RT.get(runtime.key(ob)) is not rt:
                raise BakeError("The chains changed during the bake.")
            if progress is not None:
                progress(k, total)
        missing = [f for f in range(start, end + 1) if f not in rt.record]
        if missing:
            raise BakeError("Frame %d was not simulated." % missing[0])
        Q = np.stack([rt.record[f] for f in range(start, end + 1)])
        return list(rt.sim_names), Q
    finally:
        rt_now = runtime.RT.get(runtime.key(ob))
        if rt_now is not None:
            rt_now.recording = False
            rt_now.record = {}
        runtime.BAKING.discard(runtime.key(ob))
        runtime.RT.pop(runtime.key(ob), None)


# ----------------------------------------------------------------------------------------------- reduce keys

def reduce_keys(x, y, tol):
    """Douglas-Peucker on one curve (frames x, values y) with the absolute value error: indices of the keys to
    keep so that linear interpolation between them reproduces every dropped key within tol. First and last
    always kept."""
    n = len(x)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        xs = x[a + 1:b]
        t = (xs - x[a]) / (x[b] - x[a])
        err = np.abs(y[a + 1:b] - (y[a] + t * (y[b] - y[a])))
        i = int(np.argmax(err))
        if err[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0]


# ------------------------------------------------------------------------------------------------ key writing

def _linear():
    global _LINEAR
    if _LINEAR is None:
        items = bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items
        _LINEAR = items['LINEAR'].value
    return _LINEAR


def _fill(fc, x, y):
    n = len(x)
    kp = fc.keyframe_points
    kp.add(n)
    co = np.empty(2 * n, dtype=np.float32)
    co[0::2] = x
    co[1::2] = y
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", np.full(n, _linear(), dtype=np.int32))
    fc.update()


def _curves(names, frames, Q, reduce_tol=None):
    """(bone name, index, x, y) for every F-curve to write; keys reduced when reduce_tol is set."""
    x_all = np.asarray(frames, dtype=np.float64)
    if EDITION == 'DEMO' and len(x_all) and x_all[-1] - x_all[0] >= DEMO_BAKE_FRAMES:   # never reached through
        keep = x_all - x_all[0] < DEMO_BAKE_FRAMES                                     # record(); see free_range
        x_all, Q = x_all[keep], np.asarray(Q)[keep]
    for b, name in enumerate(names):
        for i in range(4):
            y = Q[:, b, i]
            if reduce_tol is not None:
                k = reduce_keys(x_all, y, reduce_tol)
                yield name, i, x_all[k], y[k]
            else:
                yield name, i, x_all, y


def new_bake_action(ob, bake_id):
    act = bpy.data.actions.new("JK_%s_%s" % (ob.name, bake_id))
    act[TAG] = bake_id
    if not layered():
        act.id_root = 'OBJECT'
    return act


def write_curves_new_action(act, ob, curves):
    """Write the curves into the empty action. Returns (number of curves, number of keys, slot or None)."""
    n_fc = n_keys = 0
    if layered():
        slot = act.slots.new(id_type='OBJECT', name=ob.name)
        layer = act.layers.new(TRACK)
        strip = layer.strips.new(type='KEYFRAME')
        cb = strip.channelbags.new(slot)
        groups = {}
        for name, i, x, y in curves:
            fc = cb.fcurves.new(bone_path(name), index=i)
            g = groups.get(name)
            if g is None:
                g = groups[name] = cb.groups.new(name)
            fc.group = g
            _fill(fc, x, y)
            n_fc += 1
            n_keys += len(x)
        return n_fc, n_keys, slot
    for name, i, x, y in curves:
        fc = act.fcurves.new(bone_path(name), index=i, action_group=name)
        _fill(fc, x, y)
        n_fc += 1
        n_keys += len(x)
    return n_fc, n_keys, None


def write_nla(ob, names, frames, Q, bake_id, reduce_tol=None):
    """New action JK_<armature>_<bake_id> on a new NLA track 'JiggleKit Bake' (strip REPLACE, influence 1)."""
    act = new_bake_action(ob, bake_id)
    track = None
    try:
        n_fc, n_keys, slot = write_curves_new_action(act, ob, _curves(names, frames, Q, reduce_tol))
        ad = ob.animation_data or ob.animation_data_create()
        track = ad.nla_tracks.new()
        track.name = TRACK
        strip = track.strips.new(TRACK, int(frames[0]), act)
        if slot is not None and hasattr(strip, "action_slot") and strip.action_slot is None:
            strip.action_slot = slot
        strip.blend_type = 'REPLACE'
        strip.influence = 1.0
        strip.extrapolation = 'NOTHING'
        return dict(action=act.name, curves=n_fc, keys=n_keys)
    except Exception:
        if track is not None:
            ob.animation_data.nla_tracks.remove(track)
        bpy.data.actions.remove(act)
        raise


def write_action(ob, names, frames, Q, reduce_tol=None, bake_id=None):
    """Target ACTION: keys go into the active action (created when there is none); rotation F-curves of the
    simulated bones are replaced, every other F-curve is kept. The action is tagged with the bake_id so that
    Clear Bake finds the keys again."""
    ad = ob.animation_data or ob.animation_data_create()
    act = ad.action
    created = act is None
    if created:
        act = bpy.data.actions.new("%sAction" % ob.name)
        if not layered():
            act.id_root = 'OBJECT'
    curves = list(_curves(names, frames, Q, reduce_tol))
    want = set(names)
    n_fc = n_keys = 0
    if layered():
        if created:
            slot = act.slots.new(id_type='OBJECT', name=ob.name)
            ad.action = act
            if getattr(ad, "action_slot", None) is None:
                ad.action_slot = slot
        else:
            slot = ad.action_slot
            if slot is None:
                slot = act.slots.new(id_type='OBJECT', name=ob.name)
                ad.action_slot = slot
        cb = None
        for layer in act.layers:
            for strip in layer.strips:
                for c in getattr(strip, "channelbags", ()):
                    if c.slot == slot:
                        cb = c
                        break
                if cb is not None:
                    break
                if cb is None and strip.type == 'KEYFRAME':
                    cb = strip.channelbags.new(slot)
                    break
            if cb is not None:
                break
        if cb is None:
            layer = act.layers.new("Layer") if not len(act.layers) else act.layers[0]
            strip = layer.strips.new(type='KEYFRAME') if not len(layer.strips) else layer.strips[0]
            cb = strip.channelbags.new(slot)
        for fc in list(cb.fcurves):
            m = ROT_PATH.match(fc.data_path)
            if m and _unescape(m.group(1)) in want:
                cb.fcurves.remove(fc)
        groups = {g.name: g for g in cb.groups}
        for name, i, x, y in curves:
            fc = cb.fcurves.new(bone_path(name), index=i)
            g = groups.get(name)
            if g is None:
                g = groups[name] = cb.groups.new(name)
            fc.group = g
            _fill(fc, x, y)
            n_fc += 1
            n_keys += len(x)
    else:
        if created:
            ad.action = act
        for fc in list(act.fcurves):
            m = ROT_PATH.match(fc.data_path)
            if m and _unescape(m.group(1)) in want:
                act.fcurves.remove(fc)
        for name, i, x, y in curves:
            fc = act.fcurves.new(bone_path(name), index=i, action_group=name)
            _fill(fc, x, y)
            n_fc += 1
            n_keys += len(x)
    if bake_id:
        act[TAG_KEYS] = bake_id
    return dict(action=act.name, curves=n_fc, keys=n_keys)


# ----------------------------------------------------------------------------------------------------- clear

def remove_baked_keys(act, names):
    """Remove the rotation_quaternion F-curves of the bones from the action (every slot / channelbag): the keys
    a Target: Action bake wrote. Returns the number of removed curves."""
    want = set(names)
    n = 0

    def baked(fc):
        m = ROT_PATH.match(fc.data_path)
        return bool(m) and m.group(2) == "rotation_quaternion" and _unescape(m.group(1)) in want

    if layered() and hasattr(act, "layers"):
        for layer in act.layers:
            for strip in layer.strips:
                for cb in getattr(strip, "channelbags", ()):
                    for fc in [f for f in cb.fcurves if baked(f)]:
                        cb.fcurves.remove(fc)
                        n += 1
    else:
        for fc in [f for f in act.fcurves if baked(f)]:
            act.fcurves.remove(fc)
            n += 1
    return n


def find_bake(ob, bake_id):
    """(track, strip, action) made by the bake with this id, or Nones."""
    ad = ob.animation_data
    if ad is None or not bake_id:
        return None, None, None
    for tr in ad.nla_tracks:
        for st in tr.strips:
            act = st.action
            if act is not None and act.get(TAG) == bake_id:
                return tr, st, act
    for act in bpy.data.actions:
        if act.get(TAG) == bake_id:
            return None, None, act
    return None, None, None


def clear(ob):
    """Remove what the current bake made: the NLA strip (the track too when it holds nothing else) and its
    action, or, for Target: Action, the baked rotation keys of the chain bones in the tagged action. Without
    that the live simulation would resume on top of the baked jiggle (the jiggle twice)."""
    jk = ob.jigglekit
    tr, st, act = find_bake(ob, jk.bake_id)
    removed = []
    if jk.bake_id:
        names = [pb.name for pb in ob.pose.bones if pb.jigglekit.chain_id]
        for a in bpy.data.actions:
            if a.get(TAG_KEYS) == jk.bake_id:
                n = remove_baked_keys(a, names)
                del a[TAG_KEYS]
                if n:
                    removed.append("%d key curves in \"%s\"" % (n, a.name))
    if st is not None:
        if len(tr.strips) == 1:
            ob.animation_data.nla_tracks.remove(tr)
            removed.append("track")
        else:
            tr.strips.remove(st)
            removed.append("strip")
    if act is not None and act.users == 0:
        bpy.data.actions.remove(act)
        removed.append("action")
    jk.bake_id = ""
    jk.use_baked = False
    return removed


def keyed_chain_bones(ob):
    """Chain bones that still have rotation keys in the armature's active action (the live jiggle is added on
    top of such keys)."""
    ad = ob.animation_data
    if ad is None or ad.action is None:
        return []
    names = [pb.name for pb in ob.pose.bones if pb.jigglekit.chain_id]
    return keyed_rotation_bones(ad.action, getattr(ad, "action_slot", None), names)


def new_bake_id():
    return ids.new_id()

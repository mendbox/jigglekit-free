# SPDX-License-Identifier: GPL-3.0-or-later
"""Colliders (spec 4.3, 5.4): Add / Remove Collider and Auto Colliders from Selected Bones.

Auto radius: vertices of the meshes deformed by this armature with weight >= collider_min_weight on the bone, in
armature rest space; keep those projecting inside the bone (0 < t < 1); R = collider_margin * median distance to
the bone axis, at most 2 * bone length. Fewer than 8 vertices: 0.25 * bone length and a warning."""
import bpy
import numpy as np

from . import ids, model
from .create_ops import write_result

MIN_VERTS = 8


def simulated_or_child(ob, pb):
    """True when the bone or one of its parents is a simulated chain bone (forbidden collider bone, spec 5.4)."""
    p = pb
    while p is not None:
        if p.jigglekit.chain_id:
            return True
        p = p.parent
    return False


class ColliderError(Exception):
    pass


def add_collider(ob, pb, shape='CAPSULE', radius=None, t0=0.0, t1=1.0):
    """New collider on a bone. Raises ColliderError when the armature cannot store it (a library override on
    Blender 5.x); nothing is written in that case (the entry is added before the bone gets an ID)."""
    jk = ob.jigglekit
    if model.override_unsaved(ob):
        raise ColliderError(model.OVERRIDE_REFUSE % ob.name)
    try:
        c = jk.colliders.add()
    except TypeError:
        raise ColliderError(model.OVERRIDE_REFUSE % ob.name)
    c.id = ids.new_id()
    c.name = pb.name[:40]
    c.bind = 'BONE'
    c.shape = shape
    c.bone_id = ids.ensure_bone_id(pb)
    c.radius = max(1e-4, radius if radius is not None else 0.25 * pb.bone.length)
    if shape == 'SPHERE':
        c.t0 = c.t1 = 0.5 if t0 == 0.0 and t1 == 1.0 else t0
    else:
        c.t0, c.t1 = t0, t1
    jk.collider_index = len(jk.colliders) - 1
    return c


def meshes_for(ob, context, active_only):
    if active_only:
        m = context.active_object if context.active_object and context.active_object.type == 'MESH' else None
        cands = [m] if m is not None else [o for o in context.selected_objects if o.type == 'MESH']
    else:
        cands = [o for o in context.scene.objects if o.type == 'MESH']
    out = []
    for o in cands:
        if any(md.type == 'ARMATURE' and md.object == ob for md in o.modifiers):
            out.append(o)
    return out


def measure_radius(ob, meshes, bone, min_weight):
    """(radius or None, vertex count) for one bone, from the rest-pose mesh coordinates."""
    h = np.array(bone.head_local)
    t = np.array(bone.tail_local)
    ab = t - h
    L2 = float(ab @ ab)
    rho = []
    for me_ob in meshes:
        vg = me_ob.vertex_groups.get(bone.name)
        if vg is None:
            continue
        me = me_ob.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        gi = vg.index
        idx = [v.index for v in me.vertices if any(g.group == gi and g.weight >= min_weight for g in v.groups)]
        if not idx:
            continue
        Mx = np.array(ob.matrix_world.inverted(), dtype=np.float64) @ np.array(me_ob.matrix_world, dtype=np.float64)
        P = co[idx] @ Mx[:3, :3].T + Mx[:3, 3]
        tt = ((P - h) @ ab) / max(L2, 1e-30)
        inside = (tt > 0.0) & (tt < 1.0)
        Q = h + tt[inside, None] * ab
        rho.extend(np.linalg.norm(P[inside] - Q, axis=1).tolist())
    if len(rho) < MIN_VERTS:
        return None, len(rho)
    return float(np.median(rho)), len(rho)


def colliders_from_bones(context, ob, names, ui):
    msgs = []
    meshes = meshes_for(ob, context, ui.target_mesh_only)
    made = []
    for n in names:
        pb = ob.pose.bones[n]
        if simulated_or_child(ob, pb):
            msgs.append("Bone %s: collider bone is simulated or is a child of a simulated bone; skipped." % n)
            continue
        L = pb.bone.length
        if not meshes:
            msgs.append("No mesh uses this armature; used 0.25 × bone length.")
            R = 0.25 * L
        else:
            med, cnt = measure_radius(ob, meshes, pb.bone, ui.collider_min_weight)
            if med is None:
                msgs.append("Bone %s: fewer than 8 measured vertices; used fallback radius." % n)
                R = 0.25 * L
            else:
                R = min(ui.collider_margin * med, 2.0 * L)
        try:
            c = add_collider(ob, pb, shape=ui.collider_shape, radius=R)
        except ColliderError as e:
            msgs.append(str(e))
            break
        made.append((c.name, round(c.radius, 5)))
    return made, list(dict.fromkeys(msgs))


def _arm(context):
    ob = context.active_object
    return ob if ob is not None and ob.type == 'ARMATURE' else None


class JIGGLEKIT_OT_collider_add(bpy.types.Operator):
    """Add a capsule collider on the active bone (whole bone, radius a quarter of its length)"""
    bl_idname = "jigglekit.collider_add"
    bl_label = "Add Collider"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = _arm(context)
        if ob is None or ob.mode != 'POSE' or context.active_pose_bone is None:
            cls.poll_message_set("Select a body bone in Pose Mode")
            return False
        return True

    def execute(self, context):
        ob = _arm(context)
        pb = context.active_pose_bone
        if simulated_or_child(ob, pb):
            msg = "Collider bone is simulated or is a child of a simulated bone"
            write_result(context.scene, dict(ok=False, op="collider_add", error=msg, messages=[], counts={}))
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        try:
            c = add_collider(ob, pb)
        except ColliderError as e:
            write_result(context.scene, dict(ok=False, op="collider_add", error=str(e), messages=[], counts={}))
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        ob.jigglekit.version += 1
        write_result(context.scene, dict(ok=True, op="collider_add", error="", messages=[],
                                         counts=dict(colliders=len(ob.jigglekit.colliders)), collider=c.name))
        return {'FINISHED'}


class JIGGLEKIT_OT_collider_remove(bpy.types.Operator):
    """Remove the active collider"""
    bl_idname = "jigglekit.collider_remove"
    bl_label = "Remove Collider"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = _arm(context)
        if ob is None or not len(ob.jigglekit.colliders):
            cls.poll_message_set("No collider to remove")
            return False
        return True

    def execute(self, context):
        ob = _arm(context)
        jk = ob.jigglekit
        i = min(jk.collider_index, len(jk.colliders) - 1)
        jk.colliders.remove(i)
        jk.collider_index = max(0, min(i, len(jk.colliders) - 1))
        jk.version += 1
        return {'FINISHED'}


class JIGGLEKIT_OT_colliders_from_bones(bpy.types.Operator):
    """Add a collider for each selected body bone, sized from the mesh around it, so soft parts do not pass
    through the body"""
    bl_idname = "jigglekit.colliders_from_bones"
    bl_label = "Auto Colliders from Selected Bones"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = _arm(context)
        if ob is None or ob.mode != 'POSE' or not context.selected_pose_bones:
            cls.poll_message_set("Select body bones in Pose Mode")
            return False
        return True

    def execute(self, context):
        ob = _arm(context)
        names = [pb.name for pb in context.selected_pose_bones if pb.id_data == ob]
        made, msgs = colliders_from_bones(context, ob, names, context.scene.jigglekit_ui)
        ob.jigglekit.version += 1
        model.rebuild(ob)
        refused = model.OVERRIDE_REFUSE % ob.name
        err = "" if made else (refused if refused in msgs else "No collider was created.")
        write_result(context.scene, dict(ok=bool(made), op="colliders_from_bones", error=err, messages=msgs,
                                         counts=dict(colliders=len(made)), colliders=made))
        for m in msgs:
            self.report({'WARNING'}, m)
        if not made:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        return {'FINISHED'}


CLASSES = (JIGGLEKIT_OT_collider_add, JIGGLEKIT_OT_collider_remove, JIGGLEKIT_OT_colliders_from_bones)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

# SPDX-License-Identifier: GPL-3.0-or-later
"""Chain, simulation, bake and maintenance operators (spec 4.3, 5.6)."""
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty

from . import bake, ids, model, presets, props, runtime
from .create_ops import active_chain, find_armature, remove_chains, write_result

MANUAL_URL = "https://mendboxstudio.com/docs/jigglekit"


def context_armature(context):
    """Armature the panel works on: the active armature, or the armature of the active mesh (panel field first,
    then its first Armature modifier)."""
    ob = context.active_object
    if ob is None:
        return None
    if ob.type == 'ARMATURE':
        return ob
    if ob.type == 'MESH':
        ui = context.scene.jigglekit_ui
        if ui.armature is not None:
            return ui.armature
        arm, _sk = find_armature(ob, "")
        return arm
    return None


def _result(context, op, ok=True, error="", messages=(), **extra):
    d = dict(ok=ok, op=op, error=error, messages=list(messages), counts=extra.pop('counts', {}))
    d.update(extra)
    write_result(context.scene, d)


class _ChainOp:
    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None or active_chain(arm) is None:
            cls.poll_message_set("No active chain")
            return False
        return True


class JIGGLEKIT_OT_remove_chain(_ChainOp, bpy.types.Operator):
    """Remove the active chain. Its bones stay (and go back to their rest rotation) unless Delete Bones is on"""
    bl_idname = "jigglekit.remove_chain"
    bl_label = "Remove Chain"
    bl_options = {'REGISTER', 'UNDO'}

    delete_bones: BoolProperty(name="Delete Bones", default=False, options={'SKIP_SAVE'},
                               description="Also delete the chain's bones from the armature")

    def execute(self, context):
        arm = context_armature(context)
        ch = active_chain(arm)
        cid, name = ch.id, ch.name
        bones = [pb.name for pb in arm.pose.bones if pb.jigglekit.chain_id == cid]
        runtime.restore_rest(arm, bones)
        for n in bones:
            arm.pose.bones[n].jigglekit.chain_id = ""
        remove_chains(arm, {cid})
        if self.delete_bones and bones:
            vl = context.view_layer
            prev, prev_mode = context.active_object, context.active_object.mode
            if prev_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            vl.objects.active = arm
            bpy.ops.object.mode_set(mode='EDIT')
            eb = arm.data.edit_bones
            for n in bones:
                b = eb.get(n)
                if b is not None:
                    eb.remove(b)
            bpy.ops.object.mode_set(mode='OBJECT')
            vl.objects.active = prev
            if prev_mode in ('POSE', 'EDIT') and prev == arm:
                bpy.ops.object.mode_set(mode=prev_mode)
        arm.jigglekit.version += 1
        model.rebuild(arm)
        _result(context, "remove_chain", chain=name, counts=dict(bones=len(bones)),
                messages=["Bones deleted." if self.delete_bones else "Bones kept."])
        return {'FINISHED'}


class JIGGLEKIT_OT_apply_preset(_ChainOp, bpy.types.Operator):
    """Set the active chain's physics to a preset"""
    bl_idname = "jigglekit.apply_preset"
    bl_label = "Apply Preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="Preset", items=[(k, presets.UI_LABELS[k], "") for k in presets.NAMES],
                         default='GENERIC')
    all_chains: BoolProperty(name="All Chains", default=False, options={'SKIP_SAVE'},
                             description="Apply to every chain of the armature")

    def invoke(self, context, _event):
        ch = active_chain(context_armature(context))
        if not self.properties.is_property_set('preset') and ch.preset in presets.NAMES:
            self.preset = ch.preset
        return self.execute(context)

    def execute(self, context):
        arm = context_armature(context)
        chains = list(arm.jigglekit.chains) if self.all_chains else [active_chain(arm)]
        for ch in chains:
            lens = [pb.bone.length for pb in arm.pose.bones if pb.jigglekit.chain_id == ch.id]
            presets.apply_to_chain(ch, self.preset, props.applying,
                                   mean_bone_length=(sum(lens) / len(lens)) if lens else None)
        arm.jigglekit.version += 1
        _result(context, "apply_preset", preset=self.preset, counts=dict(chains=len(chains)))
        return {'FINISHED'}


class JIGGLEKIT_OT_copy_chain_settings(_ChainOp, bpy.types.Operator):
    """Copy the active chain's settings to the chains of the selected bones"""
    bl_idname = "jigglekit.copy_chain_settings"
    bl_label = "Copy Settings to Selected Chains"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = context_armature(context)
        src = active_chain(arm)
        sel = context.selected_pose_bones or []
        targets = {pb.jigglekit.chain_id for pb in sel if pb.id_data == arm and pb.jigglekit.chain_id
                   and pb.jigglekit.chain_id != src.id}
        if not targets:
            _result(context, "copy_chain_settings", ok=False, error="No other chain among the selected bones.")
            self.report({'ERROR'}, "No other chain among the selected bones.")
            return {'CANCELLED'}
        n = 0
        for ch in arm.jigglekit.chains:
            if ch.id in targets:
                ids.copy_chain_settings(src, ch)
                n += 1
        arm.jigglekit.version += 1
        _result(context, "copy_chain_settings", counts=dict(chains=n))
        return {'FINISHED'}


class JIGGLEKIT_OT_reset_sim(bpy.types.Operator):
    """Restart the simulation from the current pose; still bones go back to their rest rotation"""
    bl_idname = "jigglekit.reset_sim"
    bl_label = "Reset Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None or not len(arm.jigglekit.chains):
            cls.poll_message_set("No JiggleKit chains on this armature")
            return False
        return True

    def execute(self, context):
        arm = context_armature(context)
        old = runtime.RT.pop(runtime.key(arm), None)
        bones = [pb.name for pb in arm.pose.bones if pb.jigglekit.chain_id]
        runtime.restore_rest(arm, bones)
        rt = model.rebuild(arm) if runtime.live(arm) else None
        if rt is not None and old is not None:
            rt.sim_ms = old.sim_ms
        _result(context, "reset_sim", counts=dict(bones=len(bones)))
        return {'FINISHED'}


class JIGGLEKIT_OT_capture_rest(bpy.types.Operator):
    """Store the current rotation of the chain bones as their rest rotation (for bones without animation)"""
    bl_idname = "jigglekit.capture_rest"
    bl_label = "Capture Rest Pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None or arm.mode != 'POSE' or not len(arm.jigglekit.chains):
            cls.poll_message_set("Armature with chains, in Pose Mode")
            return False
        return True

    def execute(self, context):
        from mathutils import Quaternion
        arm = context_armature(context)
        written = runtime.written(arm)          # bones the live simulation is holding in a swung pose
        n = changed = 0
        for pb in arm.pose.bones:
            if not pb.jigglekit.chain_id or pb.rotation_mode != 'QUATERNION':
                continue
            cur = Quaternion(pb.rotation_quaternion)
            w = written.get(pb.name)
            if w is not None:
                # The pose holds jiggle (x) rest. The jiggle is taken out, so only what the user changed since
                # the last simulated frame is captured: an untouched bone keeps its rest rotation exactly.
                old = Quaternion(pb.jigglekit.rest_q)
                wq = Quaternion(w)
                if max(abs(a - b) for a, b in zip(cur, wq)) <= 1e-6:
                    n += 1
                    continue
                cur = old @ wq.inverted() @ cur
                cur.normalize()
            if max(abs(a - b) for a, b in zip(cur, pb.jigglekit.rest_q)) > 1e-7:
                changed += 1
            pb.jigglekit.rest_q = tuple(cur)
            n += 1
        arm.jigglekit.version += 1
        _result(context, "capture_rest", counts=dict(bones=n, changed=changed))
        return {'FINISHED'}


class JIGGLEKIT_OT_repair(bpy.types.Operator):
    """Fix duplicated IDs (after duplicating bones), remove dead chain links and report broken chains"""
    bl_idname = "jigglekit.repair"
    bl_label = "Repair Chains"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None:
            cls.poll_message_set("Needs an armature")
            return False
        return True

    def execute(self, context):
        arm = context_armature(context)
        msg, det = ids.repair(arm)
        runtime.RT.pop(runtime.key(arm), None)
        if runtime.live(arm):
            model.rebuild(arm)
        notes = list(det['notes'])
        _result(context, "repair", messages=[msg] + notes[:7], counts=dict(chains=det['ok']), details=det)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class JIGGLEKIT_OT_bake(bpy.types.Operator):
    """Simulate the frame range and write the jiggle as keyframes (new NLA track, or the active action).
    The live simulation of this armature pauses while the bake is active"""
    bl_idname = "jigglekit.bake"
    bl_label = "Bake to Keyframes"
    bl_options = {'REGISTER', 'UNDO'}

    fail_at: IntProperty(default=-1000000, options={'HIDDEN', 'SKIP_SAVE'})     # tests: bake_exception

    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None or not len(arm.jigglekit.chains):
            cls.poll_message_set("No JiggleKit chains on this armature")
            return False
        if arm.jigglekit.bake_id:
            cls.poll_message_set("Bake already exists. Clear it first")
            return False
        return True

    def invoke(self, context, event):
        arm = context_armature(context)
        if arm.jigglekit.bake_target == 'ACTION':
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        arm = context_armature(context)
        sc = context.scene
        jk = arm.jigglekit
        t0 = time.perf_counter()
        start, end = bake.frame_range(sc, jk)
        err = None
        names = bake.chain_bone_names(arm)
        ad = arm.animation_data
        if not names:
            err = "No chains to bake."
        elif end < start:
            err = "Frame range is empty."
        elif jk.bake_id:
            err = "Bake already exists. Clear it first."
        elif ad is not None and ad.use_tweak_mode:
            err = "Exit NLA tweak mode first."
        elif jk.bake_target == 'NLA' and ad is not None and ad.action is not None:
            keyed = bake.keyed_rotation_bones(ad.action, getattr(ad, "action_slot", None), names)
            if keyed:
                err = ("Bones %s are keyed in the active action, which overrides NLA. Push the action down to NLA "
                       "or use Target: Action." % ", ".join(keyed[:5]))
        if err:
            _result(context, "bake", ok=False, error=err)
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        frame0 = sc.frame_current
        wm = context.window_manager
        total = end - start + 1 + jk.bake_warmup
        wm.progress_begin(0, total)
        fail_at = self.fail_at if self.fail_at > -1000000 else None
        try:
            names, Q = bake.record(sc, arm, start, end, jk.bake_warmup,
                                   progress=lambda k, _n: wm.progress_update(k), fail_at=fail_at)
            t_sim = time.perf_counter() - t0
            frames = bake.key_frames(start, end, jk.bake_step)
            Qk = Q[[f - start for f in frames]]
            tol = jk.bake_tolerance if jk.bake_reduce else None
            bid = bake.new_bake_id()
            if jk.bake_target == 'NLA':
                info = bake.write_nla(arm, names, frames, Qk, bid, tol)
                msg = "Baked %d frames of %d bones to the NLA track \"%s\"." % (len(frames), len(names), bake.TRACK)
            else:
                info = bake.write_action(arm, names, frames, Qk, tol, bake_id=bid)
                msg = "Baked %d frames of %d bones into the action \"%s\"." % (len(frames), len(names),
                                                                                 info['action'])
            jk.bake_id = bid
            jk.use_baked = True                 # live paused: the jiggle is in the keys now (spec 5.6 step 7)
        except bake.BakeError as e:
            _result(context, "bake", ok=False, error=str(e))
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        finally:
            wm.progress_end()
            sc.frame_set(frame0)
        t_all = time.perf_counter() - t0
        msgs = [msg]
        if jk.bake_target == 'ACTION':
            msgs.append("The old rotation keys of these bones were replaced. Undo restores them. Clear Bake "
                        "removes the baked keys.")
        _result(context, "bake", messages=msgs,
                counts=dict(frames=len(frames), bones=len(names), curves=info['curves'], keys=info['keys']),
                timings=dict(simulate=round(t_sim, 3), total=round(t_all, 3)), details=dict(action=info['action']))
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class JIGGLEKIT_OT_clear_bake(bpy.types.Operator):
    """Remove what Bake to Keyframes made (the NLA track and its action, or the baked rotation keys of the
    jiggle bones in your action) and resume the live simulation. Other keys are not touched"""
    bl_idname = "jigglekit.clear_bake"
    bl_label = "Clear Bake"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = context_armature(context)
        if arm is None or not arm.jigglekit.bake_id:
            cls.poll_message_set("No bake on this armature")
            return False
        return True

    def execute(self, context):
        arm = context_armature(context)
        removed = bake.clear(arm)
        if removed:
            msg = "Bake cleared (%s); live simulation resumed." % ", ".join(removed)
        else:
            msg = "Live simulation resumed."
        msgs = [msg]
        left = bake.keyed_chain_bones(arm)
        if left:              # keys the bake did not make (or an action it cannot find): never silently doubled
            warn = ("Bones %s still have rotation keys in the active action: the live jiggle is added on top of "
                    "them. Remove those keys if they are a baked jiggle." % ", ".join(left[:5]))
            msgs.append(warn)
            self.report({'WARNING'}, warn)
        _result(context, "clear_bake", messages=msgs, details=dict(keyed=left))
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class JIGGLEKIT_OT_open_manual(bpy.types.Operator):
    """Open the JiggleKit manual in the web browser"""
    bl_idname = "jigglekit.open_manual"
    bl_label = "Open Manual"

    def execute(self, _context):
        bpy.ops.wm.url_open(url=MANUAL_URL)
        return {'FINISHED'}


CLASSES = (JIGGLEKIT_OT_remove_chain, JIGGLEKIT_OT_apply_preset, JIGGLEKIT_OT_copy_chain_settings,
           JIGGLEKIT_OT_reset_sim, JIGGLEKIT_OT_capture_rest, JIGGLEKIT_OT_repair, JIGGLEKIT_OT_bake,
           JIGGLEKIT_OT_clear_bake, JIGGLEKIT_OT_open_manual)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

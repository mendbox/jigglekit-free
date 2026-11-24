# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar panels (spec 4.1, 4.4): main panel with armature, Live switch, status line and Reset; subpanels Create,
Chains, Colliders, Bake, Advanced, Result. Everything is reachable with buttons (trackpad users, no right click).

Status line (Pavel's request 2026-11-23): "Live · X ms/frame" = rolling median of the simulation time only over the
last 30 frames (runtime.MEDIAN_FRAMES); above 20 ms (runtime.SLOW_MS) the line turns red and a hint row "Slow on
this computer: use Bake" shows the Bake button."""
import json
import textwrap

import bpy

from . import model, presets, runtime
from .const import DEMO_BONE_LIMIT, EDITION, RESULT, TAB
from .create_ops import active_chain
from .ops import context_armature

STORE_URL = "https://mendboxstudio.com/jigglekit"


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB


class _Sub(_Panel):
    bl_parent_id = "JIGGLEKIT_PT_main"
    bl_options = {'DEFAULT_CLOSED'}


WRAP_MAX_LINES = 8


def wrap_width(context, indent=0):
    """Characters that fit on one label line of the sidebar (about 7 px per character at UI scale 1)."""
    region = getattr(context, "region", None)
    width = region.width if region is not None and region.width > 1 else 300
    scale = getattr(getattr(context, "preferences", None), "system", None)
    scale = getattr(scale, "ui_scale", 0.0) or 1.0          # 0 in background mode (no window)
    return max(20, int(width / (7.0 * scale)) - indent)


def wrapped(layout, context, text, icon='NONE', alert=False, indent=0):
    """A label broken into lines at the sidebar width: Blender cuts a long label in the middle ("The selected
    area... created 5."), so messages are wrapped instead. The icon goes on the first line only."""
    col = layout.column(align=True)
    col.alert = alert
    # max_lines: an exception text has no length limit, and every line is a label built on every redraw
    lines = textwrap.wrap(str(text), wrap_width(context, indent), max_lines=WRAP_MAX_LINES, placeholder=" ...")
    for i, line in enumerate(lines or [""]):
        col.label(text=line, icon=icon if i == 0 else 'BLANK1' if icon != 'NONE' else 'NONE')
    return col


def plural(n, word):
    if n == 1:
        return "1 " + {"vertices": "vertex"}.get(word, word[:-1] if word.endswith("s") else word)
    return "%s %s" % (n, word)


def draw_status(layout, arm):
    text, alert, slow = runtime.status(arm)
    row = layout.row()
    row.alert = alert
    row.label(text=text, icon='ERROR' if alert else ('PLAY' if text.startswith("Live") else 'PAUSE'))
    if slow:
        # label and button on rows of their own: in one row the label was cut ("Slow on this comput...")
        hint = layout.column(align=True)
        hint.alert = True
        hint.label(text="Slow on this computer: use Bake")
        hint.operator("jigglekit.bake", text="Bake", icon='KEYINGSET')


class JIGGLEKIT_PT_main(_Panel, bpy.types.Panel):
    bl_label = "JiggleKit"
    bl_idname = "JIGGLEKIT_PT_main"

    def draw(self, context):
        layout = self.layout
        if EDITION == 'DEMO':
            box = layout.box()
            box.label(text="Free version: %d bones" % DEMO_BONE_LIMIT, icon='INFO')
            box.operator("wm.url_open", text="Get full version").url = STORE_URL
        arm = context_armature(context)
        if arm is None:
            layout.label(text="Select an armature or a skinned mesh", icon='ARMATURE_DATA')
            return
        jk = arm.jigglekit
        capped = model.CAPPED.get(arm.name) if EDITION == 'DEMO' else None
        if capped:                     # a file with more chain bones than the Free edition simulates (spec 9.3)
            r = layout.row()
            r.alert = True
            r.label(text="Simulated %d of %d bones (Free limit %d)" % (capped[0], capped[1], DEMO_BONE_LIMIT),
                    icon='ERROR')
        if model.override_unsaved(arm):      # Blender 5.x does not store JiggleKit data of a library override
            r = layout.row()
            r.alert = True
            r.label(text="Override: changes here are not saved", icon='LIBRARY_DATA_OVERRIDE')
        row = layout.row(align=True)
        row.label(text=arm.name, icon='ARMATURE_DATA')
        row.prop(jk, "enabled", text="Live", toggle=True)
        draw_status(layout, arm)
        row = layout.row()
        row.operator("jigglekit.reset_sim", text="Reset", icon='FILE_REFRESH')
        rt = runtime.RT.get(runtime.key(arm))
        if rt is not None and rt.error:
            wrapped(layout, context, "Error: " + rt.error[:160], icon='ERROR', alert=True, indent=4)


class JIGGLEKIT_PT_create(_Sub, bpy.types.Panel):
    bl_label = "Create"
    bl_options = set()

    def draw(self, context):
        layout = self.layout
        ui = context.scene.jigglekit_ui
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(ui, "region")
        col.prop(ui, "armature")
        col.prop(ui, "name")
        col.prop(ui, "mirror_x")
        if presets.get(ui.region)['kind'] == 'CHAIN':
            col.prop(ui, "segments")
        col.prop(ui, "weight_mode")
        sub = col.column()
        sub.enabled = ui.weight_mode == 'CARVE'
        sub.prop(ui, "falloff")
        sub.prop(ui, "core")
        sub.prop(ui, "max_influences")
        col.prop(ui, "parent_mode")
        layout.operator("jigglekit.create_from_selection", icon='BONE_DATA')
        # one button per row: side by side the labels were cut ("Chains from Bo...") at the default sidebar width
        col = layout.column(align=True)
        col.operator("jigglekit.add_chains_from_bones", text="Chains from Selected Bones", icon='GROUP_BONE')
        col.operator("jigglekit.paint_weights", text="Weights for Selection", icon='WPAINT_HLT')


class JIGGLEKIT_UL_chains(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active, _propname, _index=0, _flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon='BONE_DATA')
        row.label(text=presets.UI_LABELS.get(item.preset, "Custom"))
        row.prop(item, "enabled", text="")


class JIGGLEKIT_UL_colliders(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active, _propname, _index=0, _flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False,
                 icon='MESH_CAPSULE' if item.shape == 'CAPSULE' else 'MESH_UVSPHERE')
        row.prop(item, "enabled", text="")


class JIGGLEKIT_PT_chains(_Sub, bpy.types.Panel):
    bl_label = "Chains"
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return context_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        arm = context_armature(context)
        jk = arm.jigglekit
        row = layout.row()
        row.template_list("JIGGLEKIT_UL_chains", "", jk, "chains", jk, "chain_index", rows=3)
        side = row.column(align=True)
        side.operator("jigglekit.add_chains_from_bones", text="", icon='ADD')
        side.operator("jigglekit.remove_chain", text="", icon='REMOVE')
        ch = active_chain(arm)
        if ch is None:
            layout.label(text="No chains yet: use Create", icon='INFO')
            return
        row = layout.row(align=True)
        row.label(text="Preset: " + presets.UI_LABELS.get(ch.preset, "Custom"))
        row.operator_menu_enum("jigglekit.apply_preset", "preset", text="Apply", icon='PRESET')
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        for k in ("stiffness", "damping", "inertia", "gravity", "angle_limit", "tip_ratio", "tip_damping", "buckle",
                  "radius", "collide", "influence"):
            col.prop(ch, k)
        row = layout.row(align=True)
        row.operator("jigglekit.copy_chain_settings", text="Copy to Selected", icon='COPYDOWN')
        row.operator("jigglekit.remove_chain", text="Remove", icon='X')


class JIGGLEKIT_PT_colliders(_Sub, bpy.types.Panel):
    bl_label = "Colliders"

    @classmethod
    def poll(cls, context):
        return context_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        arm = context_armature(context)
        jk = arm.jigglekit
        row = layout.row()
        row.template_list("JIGGLEKIT_UL_colliders", "", jk, "colliders", jk, "collider_index", rows=3)
        side = row.column(align=True)
        side.operator("jigglekit.collider_add", text="", icon='ADD')
        side.operator("jigglekit.collider_remove", text="", icon='REMOVE')
        if len(jk.colliders):
            c = jk.colliders[min(jk.collider_index, len(jk.colliders) - 1)]
            col = layout.column()
            col.use_property_split = True
            col.use_property_decorate = False
            col.prop(c, "bind")
            if c.bind == 'BONE':
                col.prop(c, "shape")
                col.prop_search(c, "bone_name", arm.pose, "bones", text="Bone")
            else:
                col.prop(c, "object")
            col.prop(c, "radius")
            col.prop(c, "t0")
            if c.bind == 'BONE' and c.shape == 'CAPSULE':
                col.prop(c, "t1")
            col.prop(c, "offset")
        layout.prop(jk, "show_colliders")
        ui = context.scene.jigglekit_ui
        box = layout.box()
        box.operator("jigglekit.colliders_from_bones", text="Auto from Selected Bones", icon='MOD_PHYSICS')
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(ui, "collider_shape")
        col.prop(ui, "collider_margin")
        col.prop(ui, "collider_min_weight")
        col.prop(ui, "target_mesh_only")


class JIGGLEKIT_PT_bake(_Sub, bpy.types.Panel):
    bl_label = "Bake"

    @classmethod
    def poll(cls, context):
        return context_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        jk = context_armature(context).jigglekit
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(jk, "bake_use_scene_range")
        sub = col.column(align=True)
        sub.enabled = not jk.bake_use_scene_range
        sub.prop(jk, "bake_start")
        sub.prop(jk, "bake_end")
        col.prop(jk, "bake_step")
        col.prop(jk, "bake_target")
        col.prop(jk, "bake_warmup")
        col.prop(jk, "bake_reduce")
        sub = col.column()
        sub.enabled = jk.bake_reduce
        sub.prop(jk, "bake_tolerance")
        # one button per row: side by side "Bake to Keyframes" was cut ("Bake to Keyfra...") at the default width
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("jigglekit.bake", icon='KEYINGSET')
        col.operator("jigglekit.clear_bake", icon='TRASH')
        if jk.bake_id:
            layout.label(text="Baked: live simulation paused", icon='CHECKMARK')
        else:
            layout.label(text="Not baked", icon='INFO')


class JIGGLEKIT_PT_advanced(_Sub, bpy.types.Panel):
    bl_label = "Advanced"

    @classmethod
    def poll(cls, context):
        return context_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        jk = context_armature(context).jigglekit
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(jk, "substeps")
        col.prop(jk, "max_catchup")
        col.prop(jk, "time_scale")
        col.prop(jk, "gravity_vector")
        col.prop(jk, "use_cache")
        sub = col.column()
        sub.enabled = jk.use_cache
        sub.prop(jk, "cache_max_frames")
        layout.operator("jigglekit.repair", icon='TOOL_SETTINGS')
        layout.operator("jigglekit.capture_rest", icon='ARMATURE_DATA')
        layout.operator("jigglekit.open_manual", icon='HELP')


class JIGGLEKIT_PT_result(_Sub, bpy.types.Panel):
    bl_label = "Result"
    bl_options = set()

    @classmethod
    def poll(cls, context):
        if getattr(context.scene, RESULT, ""):
            return True
        arm = context_armature(context)
        rt = runtime.RT.get(runtime.key(arm)) if arm is not None else None
        return rt is not None and bool(rt.error or rt.notes)        # handler errors and rebuild notes

    def draw(self, context):
        layout = self.layout
        try:
            res = json.loads(getattr(context.scene, RESULT, "") or "{}")
        except ValueError:
            res = {}
        box = layout.box()
        box.label(text=res.get('op', "").replace("_", " ").capitalize() or "Last run",
                  icon='CHECKMARK' if res.get('ok') else 'ERROR')
        if res.get('error'):
            wrapped(box, context, res['error'], alert=True, indent=3)
        counts = res.get('counts') or {}
        if counts:
            wrapped(box, context, ", ".join(plural(v, k) for k, v in counts.items()), indent=3)
        msgs = list(res.get('messages') or [])
        arm = context_armature(context)
        rt = runtime.RT.get(runtime.key(arm)) if arm is not None else None
        if rt is not None:
            if rt.error:
                msgs.insert(0, "Simulation error (%d frames): %s" % (rt.errors, rt.error))
            msgs += [n for n in rt.notes if n not in msgs]
        for m in msgs[:8]:
            alert = res.get('ok') is False or m.startswith(("Bone", "Chain", "Collider", "Orphan", "Simulation",
                                                             "Non-uniform", "Frame rate", "Free version"))
            wrapped(box, context, m, icon='DOT', alert=alert, indent=6)
        if res.get('op') == "add_chains_from_bones" and res.get('euler'):
            # one click instead of a trip to the N panel for every FK control of a Rigify / CloudRig rig
            box.operator("jigglekit.add_chains_from_bones", text="Convert to Quaternion and Add Chains",
                         icon='GROUP_BONE').convert_rotation = True
        if res.get('op') == "add_chains_from_bones" and res.get('inherit_scale'):
            # only for bones whose Inherit Scale mode differs from Full under a scaled parent (issue E)
            box.operator("jigglekit.add_chains_from_bones", text="Set Inherit Scale to Full and Add Chains",
                         icon='GROUP_BONE').fix_inherit_scale = True


CLASSES = (JIGGLEKIT_UL_chains, JIGGLEKIT_UL_colliders, JIGGLEKIT_PT_main, JIGGLEKIT_PT_create,
           JIGGLEKIT_PT_chains, JIGGLEKIT_PT_colliders, JIGGLEKIT_PT_bake, JIGGLEKIT_PT_advanced,
           JIGGLEKIT_PT_result)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

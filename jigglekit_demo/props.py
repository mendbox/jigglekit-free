# SPDX-License-Identifier: GPL-3.0-or-later
"""Stored properties (spec 4.2, 6.1): JigglekitUISettings (Scene.jigglekit_ui), JigglekitArmature (Object.jigglekit)
with the collections JigglekitChain and JigglekitCollider, JigglekitBone (PoseBone.jigglekit) and the result string.

Every property that changes the simulation has an update function that bumps Object.jigglekit.version; the
runtime compares that number every frame and rebuilds (spec 5.5, 5.8). Editing a chain's physics value sets its
preset to CUSTOM (spec 4.2), except while presets.apply_to_chain() writes the values."""
import math

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)

from . import presets
from .const import RESULT

OV = {'LIBRARY_OVERRIDABLE'}
OV_LIST = {'LIBRARY_OVERRIDABLE', 'USE_INSERTION'}
SCHEMA = 1
_APPLYING = [0]          # > 0 while a preset writes chain values (no CUSTOM switch)


def owner(self):
    """Object that holds this property group (Object.jigglekit or an element of its collections)."""
    ob = self.id_data
    return ob if isinstance(ob, bpy.types.Object) else None


def bump(self, _context=None):
    ob = owner(self)
    if ob is not None:
        ob.jigglekit.version += 1
        from . import runtime
        runtime.invalidate()


def bump_custom(self, context=None):
    if not _APPLYING[0] and self.preset != 'CUSTOM':
        _APPLYING[0] += 1
        try:
            self.preset = 'CUSTOM'
        finally:
            _APPLYING[0] -= 1
    bump(self, context)


def cache_max_update(self, _context=None):
    """Cache size applies at once (no rebuild: the cached frames stay valid)."""
    from . import runtime
    ob = owner(self)
    rt = runtime.RT.get(runtime.key(ob)) if ob is not None else None
    if rt is not None:
        rt.cache_max = self.cache_max_frames


def redraw_views(_self=None, _context=None):
    wm = getattr(bpy.context, "window_manager", None)
    for win in (wm.windows if wm else ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class applying:
    """with props.applying(): ... writes chain values without switching the preset to CUSTOM."""

    def __enter__(self):
        _APPLYING[0] += 1

    def __exit__(self, *_exc):
        _APPLYING[0] -= 1


PRESET_ITEMS = [(k, presets.UI_LABELS[k], "Preset " + presets.UI_LABELS[k]) for k in presets.NAMES] + \
    [('CUSTOM', "Custom", "Values edited after the last preset")]
REGION_ITEMS = [(k, presets.UI_LABELS[k], "") for k in presets.NAMES]


# ------------------------------------------------------------------------------------------------- bone identity

class JigglekitBone(bpy.types.PropertyGroup):
    chain_id: StringProperty(name="Chain ID", description="Chain the bone belongs to. Empty = not simulated",
                             options={'HIDDEN'}, override=OV)
    bone_id: StringProperty(name="Bone ID", description="Stable bone ID", options={'HIDDEN'}, override=OV)
    index: IntProperty(name="Index", description="Position in the chain, 0 = root", default=0, min=0,
                       options={'HIDDEN'}, override=OV)
    rest_q: FloatVectorProperty(name="Rest Rotation", size=4, default=(1.0, 0.0, 0.0, 0.0),
                                description="Authored rotation of the bone when its rotation is not animated",
                                options={'HIDDEN'}, override=OV)


# --------------------------------------------------------------------------------------------------------- chain

class JigglekitChain(bpy.types.PropertyGroup):
    id: StringProperty(name="ID", description="Internal ID. Do not edit", options={'HIDDEN'}, override=OV)
    name: StringProperty(name="Name", description="Display name", default="Chain", maxlen=40, override=OV)
    enabled: BoolProperty(name="Enabled", description="Simulate this chain", default=True, update=bump,
                          override=OV)
    preset: EnumProperty(name="Preset", description="Last preset applied. Changing a value below sets it to Custom",
                         items=PRESET_ITEMS, default='GENERIC', override=OV)
    stiffness: FloatProperty(name="Stiffness", default=0.30, min=0.0, max=1.0, update=bump_custom, override=OV,
                             description="How strongly the bone is pulled back to the animated pose. 0 = free "
                                         "swing, 1 = very stiff and fast")
    damping: FloatProperty(name="Damping", default=0.14, min=0.0, max=1.0, update=bump_custom, override=OV,
                           description="Motion loss. Frame-rate independent. 0 = never stops, 1 = no swing")
    inertia: FloatProperty(name="Inertia", default=0.90, min=0.0, max=1.0, update=bump_custom, override=OV,
                           description="How much of the parent's motion the bone does not inherit. 0 = rigid "
                                       "follow, 1 = full lag")
    gravity: FloatProperty(name="Gravity", default=0.0, min=0.0, max=3.0, update=bump_custom, override=OV,
                           description="Multiplier of the armature gravity vector")
    angle_limit: FloatProperty(name="Angle Limit", subtype='ANGLE', default=math.radians(30.0),
                               min=math.radians(1.0), max=math.radians(170.0), update=bump_custom, override=OV,
                               description="Largest angle between the simulated bone and its animated direction")
    tip_ratio: FloatProperty(name="Tip Ratio", default=1.0, min=0.1, max=3.0, update=bump_custom, override=OV,
                             description="Stiffness of the last bone divided by stiffness of the first bone, "
                                         "linear in between")
    tip_damping: FloatProperty(name="Tip Damping", default=1.0, min=0.1, max=10.0, update=bump_custom, override=OV,
                               description="How much faster the last bone loses its motion than the first bone, "
                                           "linear in between. Above 1 keeps the end of a soft chain from "
                                           "whipping and folding")
    buckle: FloatProperty(name="Buckle", default=0.0, min=0.0, max=1.0, update=bump_custom, override=OV,
                          description="How much a bone that points up bends under a load along its length: its "
                                      "own weight, a hop, a landing. 0 = a rigid pendulum, which ignores them")
    radius: FloatProperty(name="Radius", subtype='DISTANCE', default=0.0, min=0.0, max=10.0, update=bump,
                          override=OV, description="Collision radius at each bone tail. 0 = a point")
    collide: BoolProperty(name="Collide", default=True, update=bump, override=OV,
                          description="Use the armature's colliders")
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0, update=bump, override=OV,
                             description="Mix between the animated pose (0) and the simulated pose (1)")


# ------------------------------------------------------------------------------------------------------ collider

def _bone_name_get(self):
    ob = owner(self)
    if ob is None or ob.pose is None or not self.bone_id:
        return ""
    for pb in ob.pose.bones:
        if pb.jigglekit.bone_id == self.bone_id:
            return pb.name
    return ""


def _bone_name_set(self, value):
    ob = owner(self)
    if ob is None or ob.pose is None:
        return
    pb = ob.pose.bones.get(value)
    if pb is None:
        self.bone_id = ""
    else:
        from . import ids
        self.bone_id = ids.ensure_bone_id(pb)
    bump(self)


class JigglekitCollider(bpy.types.PropertyGroup):
    id: StringProperty(name="ID", description="Internal ID", options={'HIDDEN'}, override=OV)
    name: StringProperty(name="Name", description="Display name", default="Collider", maxlen=40, override=OV)
    enabled: BoolProperty(name="Enabled", description="Use this collider", default=True, update=bump, override=OV)
    shape: EnumProperty(name="Shape", default='CAPSULE', update=bump, override=OV,
                        description="Collider shape. Object colliders support spheres only",
                        items=[('SPHERE', "Sphere", "A ball around one point"),
                               ('CAPSULE', "Capsule", "A rounded cylinder along the bone")])
    bind: EnumProperty(name="Bind", default='BONE', update=bump, override=OV,
                       description="What the collider is attached to",
                       items=[('BONE', "Bone", "Follow a bone of this armature"),
                              ('OBJECT', "Object", "Follow an object (an Empty works well); sphere only")])
    bone_id: StringProperty(name="Bone ID", description="bone_id of the bound pose bone", options={'HIDDEN'},
                            update=bump, override=OV)
    bone_name: StringProperty(name="Bone", get=_bone_name_get, set=_bone_name_set,
                              description="Bound bone. Shows the current bone name and stores its ID, so renaming "
                                          "bones keeps the binding")
    object: PointerProperty(name="Object", type=bpy.types.Object, update=bump, override=OV,
                            description="Object for OBJECT binding (an Empty works well)")
    radius: FloatProperty(name="Radius", subtype='DISTANCE', default=0.1, min=0.0001, max=100.0, update=bump,
                          override=OV, description="Collider radius")
    t0: FloatProperty(name="Start", default=0.0, min=-1.0, max=2.0, update=bump, override=OV,
                      description="Position along the bone from head (0) to tail (1). Sphere center, or capsule "
                                  "start")
    t1: FloatProperty(name="End", default=1.0, min=-1.0, max=2.0, update=bump, override=OV,
                      description="Capsule end along the bone. Ignored for spheres")
    offset: FloatVectorProperty(name="Offset", subtype='TRANSLATION', size=3, default=(0.0, 0.0, 0.0),
                                min=-10.0, max=10.0, update=bump, override=OV,
                                description="Offset in bone-local axes (X side, Y along the bone, Z front) added "
                                            "to both capsule ends")


# ------------------------------------------------------------------------------------------------------ armature

class JigglekitArmature(bpy.types.PropertyGroup):
    # USE_INSERTION: chains and colliders can be added to a library override of a linked rig (review finding 3)
    chains: CollectionProperty(type=JigglekitChain, override=OV_LIST)
    colliders: CollectionProperty(type=JigglekitCollider, override=OV_LIST)
    schema: IntProperty(name="Schema", default=SCHEMA, options={'HIDDEN'})
    enabled: BoolProperty(name="Live", description="Run the live simulation for this armature", default=True,
                          update=bump, override=OV)
    use_baked: BoolProperty(name="Use Bake", default=False, update=bump, override=OV,
                            description="A bake is active: live simulation is paused so the jiggle is not applied "
                                        "twice")
    bake_id: StringProperty(name="Bake ID", options={'HIDDEN'}, override=OV,
                            description="ID of the bake action and NLA strip created by JiggleKit")
    version: IntProperty(name="Version", default=0, options={'HIDDEN'},
                         description="Parameter version; bumped by every edit that changes the simulation")
    substeps: IntProperty(name="Substeps", default=2, min=1, max=8, update=bump, override=OV,
                          description="Minimum simulation substeps per frame. More are added automatically for "
                                      "stiff chains and low frame rates")
    max_catchup: IntProperty(name="Max Catch-up", default=3, min=1, max=8, update=bump, override=OV,
                             description="Largest frame jump simulated as one step during playback. Larger jumps "
                                         "restart the simulation")
    time_scale: FloatProperty(name="Time Scale", default=1.0, min=0.1, max=4.0, update=bump, override=OV,
                              description="Simulation speed. 1 = real time at the scene frame rate. Applies to live "
                                          "and bake")
    gravity_vector: FloatVectorProperty(name="Gravity", subtype='ACCELERATION', size=3, default=(0.0, 0.0, -9.81),
                                        min=-100.0, max=100.0, update=bump, override=OV,
                                        description="World gravity in m/s². Converted by the scene unit scale")
    use_cache: BoolProperty(name="Use Cache", default=True, update=bump, override=OV,
                            description="Remember the result of played frames so scrubbing back is instant")
    cache_max_frames: IntProperty(name="Cache Frames", default=5000, min=100, max=100000, override=OV,
                                  update=cache_max_update,
                                  description="Largest number of cached frames per armature")
    show_colliders: BoolProperty(name="Show in Viewport", default=False, override=OV, update=redraw_views,
                                 description="Draw colliders in the 3D view")
    chain_index: IntProperty(name="Active Chain", default=0, min=0)
    collider_index: IntProperty(name="Active Collider", default=0, min=0)
    bake_use_scene_range: BoolProperty(name="Scene Range", default=True, override=OV,
                                       description="Bake the scene frame range")
    bake_start: IntProperty(name="Start", default=1, min=0, max=1048574, override=OV,
                            description="First baked frame when Scene Range is off")
    bake_end: IntProperty(name="End", default=250, min=0, max=1048574, override=OV,
                          description="Last baked frame when Scene Range is off")
    bake_step: IntProperty(name="Step", default=1, min=1, max=8, override=OV,
                           description="Key every Nth frame. The simulation still runs every frame")
    bake_target: EnumProperty(name="Target", default='NLA', override=OV,
                              items=[('NLA', "NLA Track", "New NLA track with a strip"),
                                     ('ACTION', "Active Action", "Write keys into the active action (replaces "
                                                                 "rotation keys of simulated bones)")])
    bake_warmup: IntProperty(name="Warm-up", default=0, min=0, max=600, override=OV,
                             description="Frames simulated before the first baked frame, to settle loops")
    bake_reduce: BoolProperty(name="Reduce Keys", default=False, override=OV,
                              description="Remove keys that linear interpolation reproduces within the tolerance")
    bake_tolerance: FloatProperty(name="Tolerance", default=0.002, min=0.00001, max=0.1, precision=5,
                                  override=OV,
                                  description="Largest quaternion component error allowed when reducing keys")


# ---------------------------------------------------------------------------------------------- create settings

def _region_update(self, _context):
    p = presets.get(self.region)
    self.mirror_x = bool(p['mirror'])
    self.segments = int(p['bones'])


def _armature_poll(_self, ob):
    return ob.type == 'ARMATURE'


class JigglekitUISettings(bpy.types.PropertyGroup):
    region: EnumProperty(name="Region", items=REGION_ITEMS, default='CHEST', update=_region_update,
                         description="Region type. Sets bone count, mirroring and the physics preset")
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_armature_poll,
                              description="Armature to add bones to. Empty = the armature of the mesh's Armature "
                                          "modifier")
    name: StringProperty(name="Name", default="", maxlen=40,
                         description="Base name of the new bones. Empty = region label")
    mirror_x: BoolProperty(name="Mirror X", default=True,
                           description="Also create the mirrored side (armature X axis). Needs a symmetric mesh. "
                                       "Ignored when the selection is centered")
    segments: IntProperty(name="Segments", default=1, min=1, max=12,
                          description="Number of bones in a chain region. Compact regions always get one bone")
    weight_mode: EnumProperty(name="Weights", default='CARVE',
                              items=[('CARVE', "Carve", "Take weight from existing groups and give it to the new "
                                                        "bones; totals stay the same"),
                                     ('NONE', "None", "Create bones only")])
    falloff: FloatProperty(name="Falloff", default=0.6, min=0.0, max=2.0,
                           description="How far weights spread outside the selection, as a fraction of the region "
                                       "radius")
    core: FloatProperty(name="Core", default=0.4, min=0.05, max=1.0,
                        description="Inner ramp, as a fraction of the region radius, over which the weight rises "
                                    "from 0 to 1 inside the selection")
    weight_max: FloatProperty(name="Max Weight", default=1.0, min=0.1, max=1.0,
                              description="Highest weight the new bones can get")
    max_influences: IntProperty(name="Max Influences", default=0, min=0, max=8,
                                description="After blending, keep at most this many groups per vertex and "
                                            "normalize. 0 = do not limit (use 4 for game engines)")
    parent_mode: EnumProperty(name="Parent", default='AUTO',
                              items=[('AUTO', "Auto", "Parent is the bone with the most weight under the selection "
                                                      "(nearest bone if the mesh has no weights)"),
                                     ('ACTIVE_BONE', "Active Bone", "Use the active bone of the armature")])
    limit_ratio: FloatProperty(name="Limit", default=0.6, min=0.1, max=0.95,
                               description="Refuse selections that cover more than this share of the mesh vertices. A whole "
                                           "separate part (a hair or tail mesh weighted to one line of bones) is "
                                           "accepted for chain regions at any share")
    collider_margin: FloatProperty(name="Margin", default=0.9, min=0.3, max=1.5,
                                   description="Auto Colliders: radius = measured body radius × this")
    collider_min_weight: FloatProperty(name="Min Weight", default=0.5, min=0.1, max=0.95,
                                       description="Auto Colliders: only vertices with at least this weight on the "
                                                   "bone are measured")
    collider_shape: EnumProperty(name="Shape", default='CAPSULE',
                                 description="Auto Colliders: shape created for each selected bone",
                                 items=[('CAPSULE', "Capsule", ""), ('SPHERE', "Sphere", "")])
    target_mesh_only: BoolProperty(name="Active Mesh Only", default=False,
                                   description="Auto Colliders: measure only the active mesh instead of all meshes "
                                               "using this armature")


CLASSES = (JigglekitBone, JigglekitChain, JigglekitCollider, JigglekitArmature, JigglekitUISettings)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.PoseBone.jigglekit = PointerProperty(type=JigglekitBone, override=OV)
    bpy.types.Object.jigglekit = PointerProperty(type=JigglekitArmature, override=OV)
    bpy.types.Scene.jigglekit_ui = PointerProperty(type=JigglekitUISettings)
    setattr(bpy.types.Scene, RESULT, StringProperty(name="JiggleKit Result", default="", options={'HIDDEN'}))


def unregister():
    for owner_type, attr in ((bpy.types.Scene, RESULT), (bpy.types.Scene, "jigglekit_ui"),
                             (bpy.types.Object, "jigglekit"), (bpy.types.PoseBone, "jigglekit")):
        if hasattr(owner_type, attr):
            delattr(owner_type, attr)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

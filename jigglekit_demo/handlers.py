# SPDX-License-Identifier: GPL-3.0-or-later
"""Frame handlers (spec 5.5). Prototype: frame_change_pre writes rest_q, frame_change_post runs the step,
load_post / undo_post / redo_post reset the runtime, depsgraph_update_post clears the caches when an Action
changed."""
import bpy
from bpy.app.handlers import persistent

from . import runtime


@persistent
def on_frame_pre(scene, depsgraph=None):
    runtime.frame_pre(scene, depsgraph)


@persistent
def on_frame_post(scene, depsgraph=None):
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    runtime.frame_post(scene, depsgraph)


@persistent
def on_depsgraph(_scene, depsgraph=None):
    """Only: an Action changed -> anim_version + 1 (clears the caches). The add-on's own pose writes do not
    change Actions, so this does not fire on every played frame (check scrub_cache_anim_quiet)."""
    if depsgraph is None:
        return
    if depsgraph.id_type_updated('ACTION'):
        runtime.anim_changed()
    if depsgraph.id_type_updated('COLLECTION') or depsgraph.id_type_updated('SCENE'):
        runtime.invalidate()                   # objects added, removed or moved between collections


@persistent
def on_reset(*_args):
    runtime.rebuild_all()


_PAIRS = (
    ("frame_change_pre", on_frame_pre),
    ("frame_change_post", on_frame_post),
    ("depsgraph_update_post", on_depsgraph),
    ("load_post", on_reset),
    ("undo_post", on_reset),
    ("redo_post", on_reset),
)


def register():
    for name, fn in _PAIRS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)


def unregister():
    for name, fn in _PAIRS:
        lst = getattr(bpy.app.handlers, name)
        while fn in lst:
            lst.remove(fn)

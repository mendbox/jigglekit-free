# SPDX-License-Identifier: GPL-3.0-or-later
"""JiggleKit for Blender: bone jiggle physics with colliders. Mendbox Studio.
Physics core, live frame handler with cache, stored chains and colliders, Create Jiggle from Selection, chains from
existing bones, Weights for Selection, colliders, presets, Bake to Keyframes, sidebar panels and collider overlay.

The full and the free (demo) edition use the same property names (a demo file opens in the full version), so only
one of them can be enabled: register() refuses with a message when Object.jigglekit already exists (spec 9.3)."""
try:
    import bpy
except ImportError:          # plain python3: sim.py and collide.py are tested without Blender
    bpy = None

_MODULES = ("props", "create_ops", "colliders", "ops", "ui", "overlay", "handlers")
_STATE = {"registered": False, "refused": "", "prefs": None}
COEXIST_MESSAGE = "JiggleKit Free and JiggleKit cannot be enabled together. Disable one."
REENABLE_MESSAGE = "The other JiggleKit edition is disabled now. Disable and enable this add-on to start it."


def _refused_prefs():
    """Preferences entry of a refused edition: the message in red where the user just ticked the add-on (a print
    alone only reached the system console; the GUI pass of 2026-11-24 found nothing visible)."""
    class JIGGLEKIT_AP_refused(bpy.types.AddonPreferences):
        bl_idname = __package__ or __name__

        def draw(self, _context):
            col = self.layout.column()
            col.alert = True
            col.label(text=refused_message(), icon='ERROR')
    return JIGGLEKIT_AP_refused


def refused_message():
    """What the Preferences entry of a refused edition says NOW: this edition registered nothing, so
    Object.jigglekit exists only while the other edition is still enabled. Once the user disabled the other one,
    "cannot be enabled together" would be false, and this edition still does nothing until it is enabled again."""
    return COEXIST_MESSAGE if hasattr(bpy.types.Object, "jigglekit") else REENABLE_MESSAGE


def _drop_refused_prefs():
    if _STATE["prefs"] is not None:
        try:
            bpy.utils.unregister_class(_STATE["prefs"])
        except Exception:  # noqa: BLE001
            pass
        _STATE["prefs"] = None


def _mods():
    import importlib
    return [importlib.import_module("." + m, __name__) for m in _MODULES]


def register():
    if _STATE["registered"]:
        return
    if hasattr(bpy.types.Object, "jigglekit"):   # the other edition (or an old copy) is enabled
        _STATE["refused"] = COEXIST_MESSAGE
        print("JiggleKit:", COEXIST_MESSAGE)
        if _STATE["prefs"] is None:
            try:
                cls = _refused_prefs()
                bpy.utils.register_class(cls)
                _STATE["prefs"] = cls
            except Exception as e:  # noqa: BLE001  the message is a courtesy: never fail the refusal
                print("JiggleKit: could not show the message in Preferences:", repr(e))
        return
    _STATE["refused"] = ""
    _drop_refused_prefs()             # register() again after a refusal, with no unregister() between
    done = []
    try:
        for m in _mods():
            m.register()
            done.append(m)
    except Exception:
        for m in reversed(done):       # leave nothing half-registered
            try:
                m.unregister()
            except Exception:  # noqa: BLE001
                pass
        raise
    _STATE["registered"] = True


def unregister():
    _drop_refused_prefs()             # the message entry of a refused register()
    _STATE["refused"] = ""
    if not _STATE["registered"]:      # refused at register(): nothing of ours to remove
        return
    from . import runtime
    try:
        runtime.restore_all()         # no bone stays in the middle of a swing (it would be saved with the file)
    except Exception as e:  # noqa: BLE001
        print("JiggleKit: restore on disable failed:", repr(e))
    for m in reversed(_mods()):
        m.unregister()
    runtime.RT.clear()
    runtime.invalidate()
    _STATE["registered"] = False

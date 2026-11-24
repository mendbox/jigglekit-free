# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable IDs (spec 5.8): generation, duplicate repair after Edit-Mode Shift+D, Repair Chains, schema migration.

Shift+D on chain bones copies PoseBone.jigglekit with chain_id, bone_id and index (measured on 4.2.23, 4.5.14,
5.2.2, docs/PROTOTYPE-NOTES.md item 3). repair_duplicates() runs inside every rebuild, without dialogs:
- bones sharing a bone_id: the original is the bone with the shorter name without a numeric suffix .NNN (ties:
  alphabetical); the others get a new bone_id;
- duplicated bones that form a sub-chain of an existing chain become a new chain (new chain_id, a copy of the
  source chain's settings named "<name> copy"), indices renumbered from 0."""
import re
import secrets

SUFFIX = re.compile(r"\.\d{3,}$")


def new_id():
    return secrets.token_hex(8)


def ensure_bone_id(pb):
    jb = pb.jigglekit
    if not jb.bone_id:
        jb.bone_id = new_id()
    return jb.bone_id


def _orig_key(name):
    return (1 if SUFFIX.search(name) else 0, len(name), name)


def find_chain(ob, chain_id):
    for i, ch in enumerate(ob.jigglekit.chains):
        if ch.id == chain_id:
            return i, ch
    return -1, None


def copy_chain_settings(src, dst):
    """Copy every editable chain property except id and name."""
    from . import props
    with props.applying():
        for p in src.bl_rna.properties:
            k = p.identifier
            if k in ('rna_type', 'id', 'name') or p.is_readonly:
                continue
            setattr(dst, k, getattr(src, k))


def repair_duplicates(ob):
    """Returns (n_ids_fixed, [new chain names]). Writes only when duplicates exist."""
    pbs = ob.pose.bones
    by_bid = {}
    for pb in pbs:
        jb = pb.jigglekit
        if jb.chain_id and not jb.bone_id:
            jb.bone_id = new_id()
        if jb.bone_id:
            by_bid.setdefault(jb.bone_id, []).append(pb)
    dups = set()
    for lst in by_bid.values():
        if len(lst) < 2:
            continue
        orig = min(lst, key=lambda b: _orig_key(b.name))
        for pb in lst:
            if pb is not orig:
                pb.jigglekit.bone_id = new_id()
                dups.add(pb.name)
    new_chains = []
    if not dups:
        return 0, new_chains
    by_chain = {}
    for name in dups:
        pb = pbs[name]
        if pb.jigglekit.chain_id:
            by_chain.setdefault(pb.jigglekit.chain_id, []).append(pb)
    for cid, members in by_chain.items():
        names = {pb.name for pb in members}
        roots = [pb for pb in members if pb.parent is None or pb.parent.name not in names]
        _i, src = find_chain(ob, cid)
        for root in sorted(roots, key=lambda b: (b.jigglekit.index, b.name)):
            run = [root]
            while True:
                kids = [c for c in run[-1].children if c.name in names
                        and c.jigglekit.index == run[-1].jigglekit.index + 1]
                if len(kids) != 1:
                    break
                run.append(kids[0])
            nid = new_id()
            for k, pb in enumerate(run):
                pb.jigglekit.chain_id = nid
                pb.jigglekit.index = k
            if src is not None:
                ch = ob.jigglekit.chains.add()
                ch.id = nid
                copy_chain_settings(src, ch)
                ch.name = (src.name + " copy")[:40]
                new_chains.append(ch.name)
            names -= {pb.name for pb in run}
        # members left over (not reachable as a run) keep the duplicated chain_id and are cut as orphans by
        # model.rebuild with a warning
    return len(dups), new_chains


def repair(ob):
    """Repair Chains (spec 4.3): duplicate IDs, dead references. Returns (message, details dict)."""
    from . import model
    n_dup, new_chains = repair_duplicates(ob)
    jk = ob.jigglekit
    present = {pb.jigglekit.chain_id for pb in ob.pose.bones if pb.jigglekit.chain_id}
    # chain entries without any bone: dead references, removed
    dead = [i for i, ch in enumerate(jk.chains) if ch.id not in present]
    for i in reversed(dead):
        jk.chains.remove(i)
    jk.chain_index = min(jk.chain_index, max(0, len(jk.chains) - 1))
    # bones whose chain entry is gone: orphan bones, unlinked (they keep their pose)
    known = {ch.id for ch in jk.chains}
    orphans = 0
    for pb in ob.pose.bones:
        if pb.jigglekit.chain_id and pb.jigglekit.chain_id not in known:
            pb.jigglekit.chain_id = ""
            orphans += 1
    bone_ids = {pb.jigglekit.bone_id for pb in ob.pose.bones if pb.jigglekit.bone_id}
    no_bone = sum(1 for c in jk.colliders if c.bind == 'BONE' and c.bone_id not in bone_ids)
    no_bone += sum(1 for c in jk.colliders if c.bind == 'OBJECT' and c.object is None)
    built = model.describe(ob)
    jk.version += 1
    msg = "%d chains OK, %d duplicated IDs fixed, %d broken chains, %d colliders without bone" % (
        built['ok'], n_dup, built['broken'], no_bone)
    return msg, dict(ok=built['ok'], duplicated=n_dup, new_chains=new_chains, broken=built['broken'],
                     dead_chains=len(dead), orphan_bones=orphans, colliders_without_bone=no_bone,
                     notes=built['notes'])


def migrate(ob):
    """Schema migration (spec 6.1). v1: only the version check."""
    from .props import SCHEMA
    jk = ob.jigglekit
    if jk.schema < SCHEMA:
        jk.schema = SCHEMA
    return jk.schema

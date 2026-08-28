"""Build one Blender material per PBR texture folder.

Run inside Blender. Preserves the existing scene (camera, ground, lights,
world, render settings) and only adds/updates materials.
"""

import json
import os
import sys

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for _path in (SCRIPT_DIR, os.path.join(PROJECT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import material_scan  # noqa: E402
import importlib      # noqa: E402
importlib.reload(material_scan)

MATERIALS_ROOT = os.path.join(PROJECT, "materials")
TARGET_OBJECT = "material_object"

DATA_SLOTS = ("roughness", "normal", "metallic")   # rendered as Non-Color


# --------------------------------------------------------------------------- images

_image_cache = {}


def _abs(path):
    return os.path.normpath(os.path.abspath(path))


def load_image(path, colorspace):
    """Load (or reuse) an image datablock and force its colour space."""
    key = _abs(path)
    img = _image_cache.get(key)
    if img is None:
        for existing in bpy.data.images:
            if existing.filepath and _abs(bpy.path.abspath(existing.filepath)) == key:
                img = existing
                break
    if img is None:
        img = bpy.data.images.load(key, check_existing=True)
    _image_cache[key] = img
    if img.colorspace_settings.name != colorspace:
        img.colorspace_settings.name = colorspace
    return img


# --------------------------------------------------------------------------- material

def claim_material(rec, report):
    """Return the material datablock to write into, honouring rule 9.

    A pre-existing material of the intended name is reused only when it
    belongs to the same source folder, or when it carries no provenance at
    all (a leftover from an earlier partial run). Anything else gets a
    uniquely suffixed name so nothing unrelated is overwritten.
    """
    name = rec["material_name"]
    existing = bpy.data.materials.get(name)
    if existing is None:
        return bpy.data.materials.new(name)

    prev = existing.get("source_folder")
    if prev is None:
        report["rebuilt_preexisting"].append(name)
        return existing
    if os.path.normcase(str(prev)) == os.path.normcase(rec["source_folder"]):
        report["reused_matching"].append(name)
        return existing

    for suffix in "abcdefghijklmnopqrstuvwxyz":
        alt = "%s_%s" % (name, suffix)
        if alt not in bpy.data.materials:
            report["renamed_to_avoid_collision"].append({"wanted": name, "used": alt,
                                                         "occupied_by": str(prev)})
            rec["material_name"] = alt
            return bpy.data.materials.new(alt)
    raise RuntimeError("could not find a free name for " + name)


def build_material(rec, report):
    mat = claim_material(rec, report)
    # Only one material is ever assigned to the sphere, so the other 104 have
    # zero users and Blender would purge them when the .blend is next loaded.
    # A fake user is what makes the library actually persist in the file.
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Identical coordinate setup for every class: the sphere's own UV map,
    # through a neutral Mapping node (scale 1.0) that later stages can drive.
    texco = nt.nodes.new("ShaderNodeTexCoord")
    texco.location = (-1000, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-820, 0)
    mapping.name = "MAT_Mapping"
    mapping.label = "Mapping"
    mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(texco.outputs["UV"], mapping.inputs["Vector"])

    maps = rec["maps"]

    def tex_node(slot, y, colorspace, label):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.location = (-560, y)
        node.label = label
        node.name = "TEX_" + slot
        node.image = load_image(maps[slot], colorspace)
        node.interpolation = "Linear"
        node.extension = "REPEAT"
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    # ---- base colour
    if "base_color" in maps:
        n = tex_node("base_color", 320, "sRGB", "Base Color")
        nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])

    # ---- roughness
    if "roughness" in maps:
        n = tex_node("roughness", 40, "Non-Color", "Roughness")
        nt.links.new(n.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.5

    # ---- normal (OpenGL preferred; no node at all when absent)
    if "normal" in maps:
        n = tex_node("normal", -240, "Non-Color", "Normal (OpenGL)")
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-260, -240)
        nm.inputs["Strength"].default_value = 1.0
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    # ---- metallic: map if one truly exists, otherwise class default
    if "metallic" in maps:
        n = tex_node("metallic", -520, "Non-Color", "Metallic")
        nt.links.new(n.outputs["Color"], bsdf.inputs["Metallic"])
        metallic_source = "map"
    else:
        bsdf.inputs["Metallic"].default_value = \
            material_scan.METALLIC_DEFAULT[rec["dataset_class"]]
        metallic_source = "class_default"

    # ---- provenance (rule 4)
    mat["dataset_class"] = rec["dataset_class"]
    mat["source_folder"] = rec["source_folder"]
    mat["base_material_id"] = rec["base_material_id"]
    mat["source_material_name"] = rec["source_material_name"]
    mat["base_color_file"] = os.path.basename(maps.get("base_color", "")) or ""
    mat["roughness_file"] = os.path.basename(maps.get("roughness", "")) or ""
    mat["normal_file"] = os.path.basename(maps.get("normal", "")) or ""
    mat["metallic_file"] = os.path.basename(maps.get("metallic", "")) or ""
    mat["normal_variant"] = rec["normal_variant"] or "none"
    mat["metallic_source"] = metallic_source
    mat["has_displacement"] = "displacement" in maps
    mat["has_ao"] = "ao" in maps
    mat["missing_maps"] = ",".join(rec["missing_maps"])
    mat["used_default_roughness"] = bool(rec["used_default_roughness"])
    mat["needs_review"] = bool(rec["needs_review"])

    rec["metallic_source"] = metallic_source
    rec["blender_material_name"] = mat.name
    return mat


# --------------------------------------------------------------------------- main

def main():
    report = {
        "rebuilt_preexisting": [],
        "reused_matching": [],
        "renamed_to_avoid_collision": [],
        "failed": [],
    }

    records, ignored = material_scan.scan_root(MATERIALS_ROOT)
    report["ignored_dirs"] = ignored
    report["folders_found"] = len(records)

    built = []
    for rec in records:
        try:
            build_material(rec, report)
            built.append(rec)
        except Exception as exc:                       # keep going on one bad folder
            report["failed"].append({"material": rec["material_name"],
                                     "folder": rec["source_folder"],
                                     "error": "%s: %s" % (type(exc).__name__, exc)})
    report["built"] = len(built)

    # Assign the first successfully built material to the single test sphere.
    obj = bpy.data.objects.get(TARGET_OBJECT)
    if obj is not None and built:
        first = bpy.data.materials[built[0]["blender_material_name"]]
        if not obj.data.materials:
            obj.data.materials.append(first)
        else:
            obj.data.materials[0] = first
        report["assigned_to_target"] = first.name

    # Keep textures linked externally, as relative paths inside the project.
    master = os.path.join(PROJECT, "blender", "material_dataset_master.blend")
    bpy.ops.wm.save_as_mainfile(filepath=master)
    try:
        bpy.ops.file.make_paths_relative()
    except Exception as exc:
        report["make_paths_relative_error"] = str(exc)
    bpy.ops.wm.save_as_mainfile(filepath=master)
    report["blend_path"] = master

    scratch = os.path.join(PROJECT, "scripts", "_build_records.json")
    with open(scratch, "w") as fh:
        json.dump({"records": built, "report": report}, fh, indent=1)

    return report


# Guarded: importing this module must NEVER rebuild the library or save a .blend.
# It is exec'd with __name__ == "__main__" by the runner, and imported plainly
# (e.g. to reuse build_material) by other scripts.
if __name__ == "__main__":
    result = main()

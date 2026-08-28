"""Validation checks, validation renders and manifest export.

Run inside Blender, after build_materials.py.
"""

import csv
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
RENDER_ROOT = os.path.join(PROJECT, "validation_renders")
TARGET_OBJECT = "material_object"

CLASS_ORDER = material_scan.CLASSES


def dataset_materials():
    """Every material this pipeline owns, in canonical class order."""
    mats = [m for m in bpy.data.materials if m.get("base_material_id") is not None]
    def key(m):
        cls = m.get("dataset_class", "")
        rank = CLASS_ORDER.index(cls) if cls in CLASS_ORDER else 99
        return (rank, str(m.get("base_material_id", "")))
    return sorted(mats, key=key)


def _tex_nodes(mat):
    if not mat.node_tree:
        return {}
    out = {}
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.name.startswith("TEX_"):
            out[n.name[4:]] = n
    return out


def _bsdf(mat):
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def _linked(bsdf, socket):
    return bool(bsdf.inputs[socket].links)


# --------------------------------------------------------------------------- checks

def run_checks():
    res = {
        "per_class_count": {},
        "broken_file_paths": [],
        "missing_base_color": [],
        "missing_roughness": [],
        "missing_normal": [],
        "dx_normal_chosen_over_gl": [],
        "wrong_colorspace_data": [],
        "wrong_colorspace_basecolor": [],
        "metal_metallic_not_one": [],
        "nonmetal_metallic_not_zero": [],
        "default_roughness_used": [],
        "needs_manual_review": [],
        "mapping_inconsistent": [],
        "normal_strength_not_one": [],
        "total": 0,
    }

    for mat in dataset_materials():
        res["total"] += 1
        cls = mat.get("dataset_class", "?")
        res["per_class_count"][cls] = res["per_class_count"].get(cls, 0) + 1

        tex = _tex_nodes(mat)
        bsdf = _bsdf(mat)

        # 2. every image node resolves to a file that exists on disk
        for slot, node in tex.items():
            if node.image is None:
                res["broken_file_paths"].append([mat.name, slot, "<no image>"])
                continue
            abspath = bpy.path.abspath(node.image.filepath)
            if not os.path.isfile(abspath):
                res["broken_file_paths"].append([mat.name, slot, node.image.filepath])

        # 3/4/5. required maps present and actually connected
        if "base_color" not in tex or not _linked(bsdf, "Base Color"):
            res["missing_base_color"].append(mat.name)
        if "roughness" not in tex or not _linked(bsdf, "Roughness"):
            res["missing_roughness"].append(mat.name)
        if "normal" not in tex or not _linked(bsdf, "Normal"):
            res["missing_normal"].append(mat.name)

        # 6. an OpenGL normal must win whenever one exists in the folder
        folder = str(mat.get("source_folder", ""))
        if os.path.isdir(folder):
            variants = set()
            for fn in os.listdir(folder):
                hit = material_scan.classify_file(fn)
                if hit and hit[0] == "normal":
                    variants.add(hit[2])
            chosen = str(mat.get("normal_variant", "none"))
            if "gl" in variants and chosen != "gl":
                res["dx_normal_chosen_over_gl"].append([mat.name, chosen, sorted(map(str, variants))])

        # 7/8. colour-space discipline
        for slot in ("roughness", "normal", "metallic"):
            node = tex.get(slot)
            if node and node.image and node.image.colorspace_settings.name != "Non-Color":
                res["wrong_colorspace_data"].append(
                    [mat.name, slot, node.image.colorspace_settings.name])
        node = tex.get("base_color")
        if node and node.image and node.image.colorspace_settings.name != "sRGB":
            res["wrong_colorspace_basecolor"].append(
                [mat.name, node.image.colorspace_settings.name])

        # 9/10. metallic values
        metallic_mapped = _linked(bsdf, "Metallic")
        mval = bsdf.inputs["Metallic"].default_value
        if cls == "metal":
            if not metallic_mapped and abs(mval - 1.0) > 1e-6:
                res["metal_metallic_not_one"].append([mat.name, mval])
        else:
            if not metallic_mapped and abs(mval) > 1e-6:
                res["nonmetal_metallic_not_zero"].append([mat.name, mval])

        if mat.get("used_default_roughness"):
            res["default_roughness_used"].append(mat.name)
        if mat.get("needs_review"):
            res["needs_manual_review"].append(mat.name)

        # consistency extras: identical mapping across classes, normal strength 1.0
        mp = mat.node_tree.nodes.get("MAT_Mapping")
        if mp is None:
            res["mapping_inconsistent"].append([mat.name, "no mapping node"])
        else:
            sc = tuple(round(v, 6) for v in mp.inputs["Scale"].default_value)
            lo = tuple(round(v, 6) for v in mp.inputs["Location"].default_value)
            ro = tuple(round(v, 6) for v in mp.inputs["Rotation"].default_value)
            if sc != (1.0, 1.0, 1.0) or lo != (0.0, 0.0, 0.0) or ro != (0.0, 0.0, 0.0):
                res["mapping_inconsistent"].append([mat.name, str(sc), str(lo), str(ro)])
        for n in mat.node_tree.nodes:
            if n.type == "NORMAL_MAP" and abs(n.inputs["Strength"].default_value - 1.0) > 1e-6:
                res["normal_strength_not_one"].append(
                    [mat.name, n.inputs["Strength"].default_value])

    return res


# --------------------------------------------------------------------------- renders

def select_validation_set(per_class=2):
    picked = {}
    for mat in dataset_materials():
        cls = mat.get("dataset_class", "?")
        bucket = picked.setdefault(cls, [])
        if len(bucket) < per_class:
            bucket.append(mat.name)
    return picked


def render_validation(class_names, per_class=2):
    """Render the validation picks for the given classes. Camera/lights untouched."""
    obj = bpy.data.objects[TARGET_OBJECT]
    scene = bpy.context.scene
    original_slot = obj.data.materials[0].name if obj.data.materials else None
    original_output = scene.render.filepath

    picked = select_validation_set(per_class)
    done = {}
    try:
        for cls in class_names:
            for mat_name in picked.get(cls, []):
                out_dir = os.path.join(RENDER_ROOT, cls)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, mat_name + ".png")
                obj.data.materials[0] = bpy.data.materials[mat_name]
                scene.render.filepath = out_path
                bpy.ops.render.render(write_still=True)
                done[mat_name] = out_path
    finally:
        scene.render.filepath = original_output
        if original_slot and obj.data.materials:
            obj.data.materials[0] = bpy.data.materials[original_slot]
    return done


def restore_first_material():
    """Rule 11: the first successfully imported material ends up on the sphere."""
    obj = bpy.data.objects[TARGET_OBJECT]
    mats = dataset_materials()
    first = bpy.data.materials.get("MAT_Ceramic_01") or (mats[0] if mats else None)
    if first and obj.data.materials:
        obj.data.materials[0] = first
    return first.name if first else None


# --------------------------------------------------------------------------- manifest

FIELDS = [
    "class", "material_id", "blender_material_name", "source_material_name",
    "source_folder", "base_color_file", "roughness_file", "normal_file",
    "metallic_file", "metallic_value", "metallic_source", "normal_variant",
    "has_displacement", "has_ao", "missing_maps", "used_default_roughness",
    "needs_review", "validation_render",
]


def build_manifest():
    rows = []
    for mat in dataset_materials():
        bsdf = _bsdf(mat)
        cls = mat.get("dataset_class", "")
        render_rel = os.path.join("validation_renders", cls, mat.name + ".png")
        has_render = os.path.isfile(os.path.join(PROJECT, render_rel))
        rows.append({
            "class": cls,
            "material_id": mat.get("base_material_id", ""),
            "blender_material_name": mat.name,
            "source_material_name": mat.get("source_material_name", ""),
            "source_folder": mat.get("source_folder", ""),
            "base_color_file": mat.get("base_color_file", ""),
            "roughness_file": mat.get("roughness_file", ""),
            "normal_file": mat.get("normal_file", ""),
            "metallic_file": mat.get("metallic_file", ""),
            "metallic_value": round(float(bsdf.inputs["Metallic"].default_value), 4),
            "metallic_source": mat.get("metallic_source", ""),
            "normal_variant": mat.get("normal_variant", ""),
            "has_displacement": bool(mat.get("has_displacement", False)),
            "has_ao": bool(mat.get("has_ao", False)),
            "missing_maps": mat.get("missing_maps", ""),
            "used_default_roughness": bool(mat.get("used_default_roughness", False)),
            "needs_review": bool(mat.get("needs_review", False)),
            "validation_render": render_rel.replace("\\", "/") if has_render else "",
        })
    return rows


def write_manifest():
    rows = build_manifest()
    csv_path = os.path.join(PROJECT, "metadata", "material_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(PROJECT, "metadata", "material_manifest.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    return {"csv": csv_path, "json": json_path, "rows": len(rows)}

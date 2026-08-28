"""Headless preflight probe. Run with:
   blender --background <file.blend> --python scripts/_headless_probe.py
Prints a single JSON line prefixed with PROBE_JSON: for easy parsing.
"""
import json
import math
import os
import sys

import bpy

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}

sc = bpy.context.scene
out["file"] = bpy.data.filepath

# RenderSynth availability
try:
    import addon_utils
    rs = [m.__name__ for m in addon_utils.modules()
          if "rendersynth" in (m.__name__ or "").lower()]
    out["rendersynth_modules"] = rs
    out["rendersynth_enabled"] = bool(rs) and addon_utils.check(rs[0])[1]
except Exception as exc:
    out["rendersynth_error"] = str(exc)

out["has_pipeline_data"] = hasattr(sc, "pipeline_data")
if out["has_pipeline_data"]:
    out["pipeline_ops"] = [(o.operation_type, o.name, o.enabled, o.valid)
                           for o in sc.pipeline_data.operations]
    out["randomizer_seed"] = sc.randomizer_seed

try:
    import importlib
    ep = importlib.import_module(
        "bl_ext.user_default.rendersynth.core.executable_pipeline")
    p = ep.ExecutablePipeline(bpy.context, sc.pipeline_data)
    out["compiled_executors"] = [type(e).__name__ for e in p.get_executors()]
except Exception as exc:
    out["compile_error"] = "%s: %s" % (type(exc).__name__, exc)

ds = [m for m in bpy.data.materials if m.get("base_material_id")]
per_class = {}
for m in ds:
    per_class[m["dataset_class"]] = per_class.get(m["dataset_class"], 0) + 1
out["n_dataset_materials"] = len(ds)
out["per_class"] = per_class
out["metal_13"] = (bpy.data.materials["MAT_Metal_13"]["source_material_name"]
                   if "MAT_Metal_13" in bpy.data.materials else None)

cam = bpy.data.objects.get("Camera")
key = bpy.data.objects.get("Key_Area")
obj = bpy.data.objects.get("material_object")
out["baseline"] = {
    "cam_loc": [round(v, 6) for v in cam.location],
    "cam_rot_deg": [round(math.degrees(v), 4) for v in cam.rotation_euler],
    "lens": cam.data.lens, "dof": cam.data.dof.use_dof,
    "light_loc": [round(v, 6) for v in key.location],
    "light_power": key.data.energy, "light_size": key.data.size,
    "light_color": list(key.data.color),
    "obj_rot": [round(v, 6) for v in obj.rotation_euler],
    "obj_loc": [round(v, 6) for v in obj.location],
    "sphere_mat": obj.data.materials[0].name if obj.data.materials else None,
    "ground": [round(v, 2) for v in bpy.data.objects["Ground"].dimensions],
}
out["render"] = {
    "engine": sc.render.engine,
    "res": [sc.render.resolution_x, sc.render.resolution_y],
    "pct": sc.render.resolution_percentage,
    "fmt": sc.render.image_settings.file_format,
    "color": sc.render.image_settings.color_mode,
    "film_transparent": sc.render.film_transparent,
    "motion_blur": sc.render.use_motion_blur,
    "samples": getattr(sc.eevee, "taa_render_samples", None),
    "view_transform": sc.view_settings.view_transform,
}

# every texture must resolve on disk
broken = []
for m in ds:
    for n in m.node_tree.nodes:
        if n.type == "TEX_IMAGE":
            if n.image is None:
                broken.append([m.name, n.name, "<no image>"])
            elif not os.path.isfile(bpy.path.abspath(n.image.filepath)):
                broken.append([m.name, n.name, n.image.filepath])
out["broken_textures"] = broken

print("PROBE_JSON:" + json.dumps(out))

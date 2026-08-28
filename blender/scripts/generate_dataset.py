"""Deterministic dataset generator around the RenderSynth pipeline.

WHAT RENDERSYNTH DOES NATIVELY (compiled and executed through its own
ExecutablePipeline / OperationRegistry):

    Rotation  -> material_object   object rotation randomization
    Power     -> Key_Area          area-light power randomization
    Position  -> Key_Area          area-light position jitter (offset mode)

WHY A WRAPPER IS NEEDED (capabilities RenderSynth 1.0.0 does not provide):

  1. Per-render deterministic seeds. RenderSynth's Executor seeds Python's
     `random` ONCE per batch (core/generation.py) and then renders `amount`
     frames. The brief requires a child seed derived from
     master|class|material_id|render_index. The wrapper seeds the same global
     RNG that every RenderSynth sampler draws from, immediately before each
     render, so RenderSynth's own operations become per-render reproducible.

  2. Deterministic iteration over 105 known materials. The native Material
     operation samples randomly from a pool; it cannot walk a fixed list, and
     the split assignment must come from the manifest, not from chance.

  3. Classification directory output. WritingConfig.folder_structure is
     Literal["yolo"] (core/configurations.py) - a detection layout. The brief
     needs dataset/<split>/<class>/ and pilot_dataset/<class>/.

  4. Spherical camera orbit around a target. PipeNames.SPHERE_LOCK and
     JITTER are declared but NOT registered in OperationRegistry; the
     registered camera ops are Change POVs (discrete position pool), Bezier
     Lock (curve-following) and Focal Length. None expresses a continuous
     azimuth/elevation/distance envelope around the validated baseline.
     The wrapper does the geometry, but samples the three orbit parameters
     with RenderSynth's OWN distribution engine (SamplerCompiler /
     PresetSampler), so the randomization stays inside RenderSynth.

  5. Aiming the area light after it moves. No native "look at" operation.
"""

import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import configure_rendersynth as cr
importlib.reload(cr)

PROJECT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MANIFEST = os.path.join(PROJECT, "metadata", "material_manifest_split.csv")

TARGET_OBJECT = "material_object"
LIGHT_OBJECT = "Key_Area"
CAMERA_OBJECT = "Camera"

MASTER_SEED = cr.MASTER_SEED
CLASS_ORDER = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]

# RenderSynth's own distribution machinery
_comp = importlib.import_module("bl_ext.user_default.rendersynth.distribution.computation")
_ep = importlib.import_module("bl_ext.user_default.rendersynth.core.executable_pipeline")
SamplerCompiler = _comp.SamplerCompiler
ExecutablePipeline = _ep.ExecutablePipeline


# --------------------------------------------------------------------------- seeding

def child_seed(cls, material_id, render_index):
    """Stable per-render seed. SHA-256, not Python's salted hash()."""
    key = "%d|%s|%s|%02d" % (MASTER_SEED, cls, material_id, render_index)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


# --------------------------------------------------------------------------- baseline

class Baseline:
    """The validated composition, captured once and restored at the end."""

    def __init__(self):
        cam = bpy.data.objects[CAMERA_OBJECT]
        light = bpy.data.objects[LIGHT_OBJECT]
        obj = bpy.data.objects[TARGET_OBJECT]

        self.target = obj.matrix_world.translation.copy()
        self.cam_location = cam.location.copy()
        self.cam_rotation = cam.rotation_euler.copy()
        self.light_location = light.location.copy()
        self.light_rotation = light.rotation_euler.copy()
        self.light_power = light.data.energy
        self.obj_rotation = obj.rotation_euler.copy()
        self.material = obj.data.materials[0].name if obj.data.materials else None
        self.render_filepath = bpy.context.scene.render.filepath

        v = self.cam_location - self.target
        self.distance = v.length
        self.azimuth = math.degrees(math.atan2(v.x, -v.y))
        self.elevation = math.degrees(math.asin(v.z / v.length))

    def restore(self):
        cam = bpy.data.objects[CAMERA_OBJECT]
        light = bpy.data.objects[LIGHT_OBJECT]
        obj = bpy.data.objects[TARGET_OBJECT]
        cam.location = self.cam_location
        cam.rotation_euler = self.cam_rotation
        light.location = self.light_location
        light.rotation_euler = self.light_rotation
        light.data.energy = self.light_power
        obj.rotation_euler = self.obj_rotation
        if self.material and obj.data.materials:
            obj.data.materials[0] = bpy.data.materials[self.material]
        bpy.context.scene.render.filepath = self.render_filepath


def look_at(location, target):
    """Euler that points an object's -Z at target with +Y up. Yields zero roll."""
    return (target - location).to_track_quat('-Z', 'Y').to_euler()


# --------------------------------------------------------------------------- camera

def _uniform(lo, hi):
    """A 1-D uniform sampler built by RenderSynth's own SamplerCompiler."""
    return SamplerCompiler.make_distribution("UNIFORM", 1, min=lo, max=hi)


class CameraOrbit:
    """Continuous orbit around the validated baseline, in RenderSynth distributions."""

    def __init__(self, baseline):
        self.baseline = baseline
        self.d_azimuth = _uniform(*cr.CAM_AZIMUTH_DELTA_DEG)
        self.d_elevation = _uniform(*cr.CAM_ELEVATION_DELTA_DEG)
        self.f_distance = _uniform(*cr.CAM_DISTANCE_FACTOR)

    def apply(self):
        b = self.baseline
        az = b.azimuth + self.d_azimuth.sample()[0]
        el = b.elevation + self.d_elevation.sample()[0]
        dist = b.distance * self.f_distance.sample()[0]

        ar, er = math.radians(az), math.radians(el)
        loc = Vector((
            b.target.x + dist * math.cos(er) * math.sin(ar),
            b.target.y - dist * math.cos(er) * math.cos(ar),
            b.target.z + dist * math.sin(er),
        ))
        cam = bpy.data.objects[CAMERA_OBJECT]
        cam.location = loc
        cam.rotation_euler = look_at(loc, b.target)
        return {"azimuth_deg": az, "elevation_deg": el, "distance": dist}


# --------------------------------------------------------------------------- manifest

def load_split_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_id"] = r.get("base_material_id") or r["material_id"]
    return rows


def select_pilot(rows):
    """Alphabetically first TRAIN material of each class - deterministic."""
    picks = []
    for cls in CLASS_ORDER:
        train = sorted((r for r in rows if r["class"] == cls and r["split"] == "train"),
                       key=lambda r: r["_id"])
        if train:
            picks.append(train[0])
    return picks


RENDER_FIELDS = [
    "image_path", "filename", "class", "material_id", "blender_material_name",
    "split", "render_index", "random_seed",
    "camera_location_x", "camera_location_y", "camera_location_z",
    "camera_rotation_x_deg", "camera_rotation_y_deg", "camera_rotation_z_deg",
    "camera_azimuth_deg", "camera_elevation_deg", "camera_distance",
    "light_location_x", "light_location_y", "light_location_z", "light_power",
    "object_rotation_x_deg", "object_rotation_y_deg", "object_rotation_z_deg",
    "render_engine", "resolution", "source_base_material",
    "material_verified_on_object",
]


# --------------------------------------------------------------------------- run

def generate(records, renders_per_material, out_root, layout, manifest_path,
             dry_run=False):
    """Render `renders_per_material` images for each record.

    layout "flat"  -> <out_root>/<class>/            (pilot)
    layout "split" -> <out_root>/<split>/<class>/    (full dataset)
    """
    scene = bpy.context.scene
    obj = bpy.data.objects[TARGET_OBJECT]
    light = bpy.data.objects[LIGHT_OBJECT]
    cam = bpy.data.objects[CAMERA_OBJECT]

    baseline = Baseline()
    orbit = CameraOrbit(baseline)
    pipeline = ExecutablePipeline(bpy.context, scene.pipeline_data)
    full_ctx = pipeline.build_context_manager()

    resolution = "%dx%d" % (scene.render.resolution_x, scene.render.resolution_y)
    rows = []

    try:
        with full_ctx:
            for rec in records:
                cls, mid = rec["class"], rec["_id"]
                mat = bpy.data.materials[rec["blender_material_name"]]

                if layout == "split":
                    out_dir = os.path.join(out_root, rec["split"], cls)
                else:
                    out_dir = os.path.join(out_root, cls)
                os.makedirs(out_dir, exist_ok=True)

                for idx in range(1, renders_per_material + 1):
                    seed = child_seed(cls, mid, idx)
                    random.seed(seed)
                    try:
                        import numpy as np
                        np.random.seed(seed % (2 ** 32))
                    except ImportError:
                        pass

                    # --- deterministic material assignment (wrapper)
                    obj.data.materials[0] = mat

                    # --- native RenderSynth operations, inside its frame context
                    with full_ctx.frame_context():
                        pipeline.execute()

                        # --- wrapper geometry: camera orbit + light aiming
                        orbit_info = orbit.apply()
                        light.rotation_euler = look_at(light.location, baseline.target)

                        filename = "%s_r%02d.png" % (mid, idx)
                        out_path = os.path.join(out_dir, filename)

                        if not dry_run:
                            scene.render.filepath = out_path
                            bpy.ops.render.render(write_still=True)

                        rows.append({
                            "image_path": os.path.relpath(out_path, PROJECT).replace("\\", "/"),
                            "filename": filename,
                            "class": cls,
                            "material_id": mid,
                            "blender_material_name": mat.name,
                            "split": rec["split"],
                            "render_index": idx,
                            "random_seed": seed,
                            "camera_location_x": round(cam.location.x, 6),
                            "camera_location_y": round(cam.location.y, 6),
                            "camera_location_z": round(cam.location.z, 6),
                            "camera_rotation_x_deg": round(math.degrees(cam.rotation_euler.x), 4),
                            "camera_rotation_y_deg": round(math.degrees(cam.rotation_euler.y), 4),
                            "camera_rotation_z_deg": round(math.degrees(cam.rotation_euler.z), 4),
                            "camera_azimuth_deg": round(orbit_info["azimuth_deg"], 4),
                            "camera_elevation_deg": round(orbit_info["elevation_deg"], 4),
                            "camera_distance": round(orbit_info["distance"], 6),
                            "light_location_x": round(light.location.x, 6),
                            "light_location_y": round(light.location.y, 6),
                            "light_location_z": round(light.location.z, 6),
                            "light_power": round(light.data.energy, 3),
                            "object_rotation_x_deg": round(math.degrees(obj.rotation_euler.x), 4),
                            "object_rotation_y_deg": round(math.degrees(obj.rotation_euler.y), 4),
                            "object_rotation_z_deg": round(math.degrees(obj.rotation_euler.z), 4),
                            "render_engine": scene.render.engine,
                            "resolution": resolution,
                            "source_base_material": rec.get("source_folder", ""),
                            # read back off the object, not assumed
                            "material_verified_on_object": obj.data.materials[0].name,
                        })
    finally:
        baseline.restore()

    if manifest_path:
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=RENDER_FIELDS)
            w.writeheader()
            w.writerows(rows)

    return rows


def run_pilot(renders_per_material=5, dry_run=False):
    rows = load_split_manifest()
    picks = select_pilot(rows)
    out_root = os.path.join(PROJECT, "pilot_dataset")
    manifest = os.path.join(PROJECT, "metadata", "pilot_render_manifest.csv")
    result_rows = generate(picks, renders_per_material, out_root, "flat", manifest,
                           dry_run=dry_run)
    return {"picked": [r["_id"] for r in picks], "rows": len(result_rows),
            "out_root": out_root, "manifest": manifest}


def run_full(renders_per_material=20, dry_run=False):
    """NOT run yet - awaiting approval."""
    rows = load_split_manifest()
    out_root = os.path.join(PROJECT, "dataset")
    manifest = os.path.join(PROJECT, "metadata", "render_manifest_full.csv")
    return generate(rows, renders_per_material, out_root, "split", manifest,
                    dry_run=dry_run)

"""Configure the RenderSynth pipeline for material-classification dataset generation.

Builds three NATIVE RenderSynth operations in scene.pipeline_data:

    Rotation  -> material_object   (texture orientation)
    Power     -> Key_Area light    (exposure variation)
    Position  -> Key_Area light    (light position jitter, offset mode)

Camera orbit is NOT built here: RenderSynth 1.0.0 registers no spherical
camera operation (PipeNames.SPHERE_LOCK / JITTER / POV-continuous are either
unregistered or pool-based), so the wrapper in generate_dataset.py handles it
using RenderSynth's own distribution samplers.

Run inside Blender.
"""

import json
import math
import os

import bpy

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_OBJECT = "material_object"
LIGHT_OBJECT = "Key_Area"

MASTER_SEED = 62642

# ---- randomization envelope (single source of truth, mirrored by the wrapper)
OBJ_ROT_X_DEG = (-10.0, 10.0)
OBJ_ROT_Y_DEG = (-10.0, 10.0)
OBJ_ROT_Z_DEG = (0.0, 360.0)

LIGHT_POWER_FACTOR = (0.80, 1.20)
LIGHT_POS_JITTER = ((-1.0, 1.0), (-1.0, 1.0), (-0.6, 0.6))   # metres, offset from baseline

CAM_AZIMUTH_DELTA_DEG = (-20.0, 20.0)
# Elevation floor is a hard geometric constraint, not a taste choice: the camera's
# half-FOV is 19.8 deg, so below that the top-of-frame ray never descends to the
# ground and a horizon band enters frame. -4 deg (21 deg absolute) keeps the top
# ray landing ~126 m out, well inside the enlarged ground plane.
CAM_ELEVATION_DELTA_DEG = (-4.0, 12.0)

# Ground plane enlarged from the validated 60 m so the -4 deg elevation stays clean.
# MAT_Ground is an untextured Principled BSDF, so this is visually neutral.
GROUND_OBJECT = "Ground"
GROUND_SIZE_M = 1000.0
CAM_DISTANCE_FACTOR = (0.94, 1.06)


def _simple_dist(preset, parameters):
    """A RenderSynth 'simple' (non node-tree) distribution config."""
    return {
        "use_tree": False,
        "preset": preset,                 # Distribution[<NAME>] - enum NAME, not value
        "do_discretize": False,
        "do_clamp": False,
        "clamping_factors": [0.0, 0.0],
        "parameters": parameters,
    }


def _axis(x=True, y=True, z=True, dims=3):
    # AxisSetMask reads WidgetSerializationKeys.DIMENSION ("dimension");
    # "dims" is carried too so either key satisfies the mask.
    return {"dimension": dims, "dims": dims,
            "randomize_x": x, "randomize_y": y, "randomize_z": z}


def build_configs(baseline_power):
    rad = math.radians
    lo_p = baseline_power * LIGHT_POWER_FACTOR[0]
    hi_p = baseline_power * LIGHT_POWER_FACTOR[1]

    rotation_cfg = {
        "dimension": 3,
        "distribution": _simple_dist("MULTIVARIATE_UNIFORM", {
            "min_vec": [rad(OBJ_ROT_X_DEG[0]), rad(OBJ_ROT_Y_DEG[0]), rad(OBJ_ROT_Z_DEG[0])],
            "max_vec": [rad(OBJ_ROT_X_DEG[1]), rad(OBJ_ROT_Y_DEG[1]), rad(OBJ_ROT_Z_DEG[1])],
        }),
        "axis": _axis(),
        "target": {"names": [TARGET_OBJECT]},
        "offset": {"mode": False},        # absolute assign; baseline rotation is (0,0,0)
    }

    power_cfg = {
        "distribution": _simple_dist("UNIFORM", {"min": lo_p, "max": hi_p}),
        # SimpleLightPropertyOperation looks the target up in bpy.data.lights
        "typed_obj": {"name": bpy.data.objects[LIGHT_OBJECT].data.name},
        "offset": {"mode": False},
    }

    light_pos_cfg = {
        "dimension": 3,
        "distribution": _simple_dist("MULTIVARIATE_UNIFORM", {
            "min_vec": [LIGHT_POS_JITTER[0][0], LIGHT_POS_JITTER[1][0], LIGHT_POS_JITTER[2][0]],
            "max_vec": [LIGHT_POS_JITTER[0][1], LIGHT_POS_JITTER[1][1], LIGHT_POS_JITTER[2][1]],
        }),
        "axis": _axis(),
        "target": {"names": [LIGHT_OBJECT]},
        "offset": {"mode": True},         # jitter relative to the validated position
    }

    return [
        ("Rotation", "Object Rotation (material_object)", rotation_cfg),
        ("Power", "Area Light Power (Key_Area)", power_cfg),
        ("Position", "Area Light Position jitter (Key_Area)", light_pos_cfg),
    ]


def enlarge_ground(size=None):
    """Widen the ground plane so low-elevation camera samples stay off the horizon."""
    size = size or GROUND_SIZE_M
    gnd = bpy.data.objects[GROUND_OBJECT]
    before = tuple(gnd.dimensions)
    if abs(gnd.dimensions.x - size) > 1e-3:
        factor = size / gnd.dimensions.x
        gnd.scale = tuple(s * factor for s in gnd.scale)
        bpy.context.view_layer.update()
    return {"before": [round(v, 2) for v in before],
            "after": [round(v, 2) for v in gnd.dimensions],
            "scale": [round(v, 4) for v in gnd.scale]}


def build_pipeline():
    scene = bpy.context.scene
    light = bpy.data.objects[LIGHT_OBJECT]
    baseline_power = light.data.energy

    pd = scene.pipeline_data
    pd.operations.clear()

    for order, (op_type, name, cfg) in enumerate(build_configs(baseline_power), start=1):
        op = pd.operations.add()
        op.operation_type = op_type
        op.name = name
        op.config = json.dumps(cfg)
        op.enabled = True
        op.order = order
        op.valid = True               # ExecutablePipeline skips operations flagged invalid

    scene.randomizer_seed = MASTER_SEED
    return len(pd.operations), baseline_power


def verify_compiles():
    """Compile the pipeline with RenderSynth's own compiler - proves the configs are valid."""
    import importlib
    ep_mod = importlib.import_module(
        "bl_ext.user_default.rendersynth.core.executable_pipeline")
    pipeline = ep_mod.ExecutablePipeline(bpy.context, bpy.context.scene.pipeline_data)
    execs = pipeline.get_executors()
    return [type(e).__name__ for e in execs]


def save_pipeline_json():
    scene = bpy.context.scene
    path = os.path.join(PROJECT, "blender", "configs", "rendersynth_material_pipeline.json")
    scene.randomizer_pipeline_save_path = path
    bpy.ops.randomizer.save_pipeline()

    # RenderSynth's serializer only records its own operations. Append the
    # wrapper-side envelope so this single file is the complete, reproducible
    # description. PipelineLoader reads only known keys, so the extra
    # "wrapper" section is inert on load.
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["wrapper"] = {
        "script": "blender/scripts/generate_dataset.py",
        "reason": "per-render seeds, deterministic material iteration, "
                  "classification output layout, spherical camera orbit, light aiming",
        "master_seed": MASTER_SEED,
        "child_seed": "sha256('<master>|<class>|<material_id>|<rXX>')[:8] % (2**31-1)",
        "camera_orbit": {
            "sampler": "rendersynth SamplerCompiler.make_distribution('UNIFORM')",
            "azimuth_delta_deg": list(CAM_AZIMUTH_DELTA_DEG),
            "elevation_delta_deg": list(CAM_ELEVATION_DELTA_DEG),
            "distance_factor": list(CAM_DISTANCE_FACTOR),
            "baseline": "captured from the scene at run time",
            "aim": "look_at target centre, -Z forward / +Y up (zero roll)",
        },
        "light_aim": "re-aimed at target centre after every Position sample",
        "ground_plane_size_m": GROUND_SIZE_M,
        "texture_mapping": "Mapping node fixed at scale 1.0 - not randomized",
        "unrandomized": ["world lighting", "light size", "light colour",
                         "texture scale", "PBR material parameters",
                         "focal length", "depth of field", "motion blur"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return path

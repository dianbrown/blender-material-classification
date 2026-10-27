"""Standalone Blender 5.1 renderer. No .blend input, saving, or V1 output writes.

Run with --factory-startup --background --python-exit-code 1 --python THIS -- ...
Only the explicit holdout mode renders fresh sources. Development uses one old
TRAIN source per class. Smoke uses one old TRAIN source, Cycles CPU, 64px, 1 sample.
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_common import (  # noqa: E402
    CLASSES,
    absolute_path,
    child_seed,
    digest,
    fingerprint,
    inside,
    read_json,
    require_suffix,
    validate_identity_audit,
    validate_png,
    validate_receipt,
    validate_registry,
    write_json,
)

FIELDS = [
    "image_path",
    "filename",
    "class",
    "material_id",
    "source_asset_id",
    "provider",
    "split",
    "original_split",
    "render_index",
    "random_seed",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_distance",
    "camera_location",
    "light_location",
    "light_power",
    "object_rotation_deg",
    "render_engine",
    "blender_version",
    "resolution_width",
    "resolution_height",
    "selection_sha256",
    "source_sha256",
    "renderer_sha256",
    "config_sha256",
    "scene_sha256",
    "run_sha256",
    "parameters_sha256",
    "image_sha256",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("holdout", "development", "smoke"))
    for name in ("config", "out", "manifest", "original-manifest"):
        parser.add_argument("--" + name, type=absolute_path, required=True)
    for name in (
        "registry",
        "receipt",
        "sources-root",
        "identity-audit",
        "development-materials-root",
        "development-manifest",
    ):
        parser.add_argument("--" + name, type=absolute_path)
    args = parser.parse_args(argv)
    required = (
        ("registry", "receipt", "sources_root", "identity_audit")
        if args.mode == "holdout"
        else ("development_materials_root", "development_manifest")
    )
    forbidden = (
        ("development_materials_root", "development_manifest")
        if args.mode == "holdout"
        else ("registry", "receipt", "sources_root", "identity_audit")
    )
    if any(getattr(args, key) is None for key in required):
        parser.error(
            f"{args.mode} requires {', '.join('--' + k.replace('_', '-') for k in required)}"
        )
    if any(getattr(args, key) is not None for key in forbidden):
        parser.error("fresh and development sources must not be mixed")
    return args


def validate_paths(args):
    leaf = {
        "holdout": "holdout_dataset",
        "development": "holdout_development",
        "smoke": "holdout_smoke",
    }[args.mode]
    require_suffix(args.out, ("artifacts", "v3", leaf))
    if args.mode == "holdout":
        require_suffix(args.sources_root, ("artifacts", "v3", "holdout_sources"))
        require_suffix(args.manifest, ("metadata", "v3_holdout", "render_manifest.csv"))
        if args.manifest.parents[2] != args.out.parents[2]:
            raise ValueError(
                "production output and manifest must belong to the same explicit project root"
            )
    elif args.manifest != args.out / "render_manifest.csv":
        raise ValueError(
            "development/smoke manifests must stay inside their isolated output directory"
        )


def validate_config(config):
    if config.get("protocol_sha256") != fingerprint(
        {k: v for k, v in config.items() if k != "protocol_sha256"}
    ):
        raise ValueError("frozen render protocol fingerprint mismatch")
    if (
        config["schema_version"] != 1
        or config["blender_version"] != [5, 1, 1]
        or config["views_per_material"] != 30
        or config["engine"] != "BLENDER_EEVEE"
        or config["resolution"] != [512, 512]
        or config["normal_convention"] != "OpenGL"
        or config["uv_scale"] != [1, 1, 1]
    ):
        raise ValueError("unsupported holdout protocol")


def parameters(config, cls, material_id, index):
    seed = child_seed(config["master_seed"], cls, material_id, index)
    rng = random.Random(seed)
    # Same independent uniforms and order as the V1 operations, without the add-on.
    rotation = [rng.uniform(*bounds) for bounds in config["object_rotation_deg"]]
    power = rng.uniform(*config["light_power"])
    light = [rng.uniform(*bounds) for bounds in config["light_location"]]
    azimuth = rng.uniform(*config["camera_azimuth_deg"])
    elevation = rng.uniform(*config["camera_elevation_deg"])
    distance = rng.uniform(*config["camera_distance"])
    ar, er = math.radians(azimuth), math.radians(elevation)
    target = config["sphere_center"]
    camera = [
        target[0] + distance * math.cos(er) * math.sin(ar),
        target[1] - distance * math.cos(er) * math.cos(ar),
        target[2] + distance * math.sin(er),
    ]
    return {
        "random_seed": seed,
        "object_rotation_deg": rotation,
        "light_power": power,
        "light_location": light,
        "camera_azimuth_deg": azimuth,
        "camera_elevation_deg": elevation,
        "camera_distance": distance,
        "camera_location": camera,
    }


def development_assets(args):
    """Use only explicitly supplied old training sources, never val/test."""
    with args.development_manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with args.original_manifest.open(encoding="utf-8-sig", newline="") as stream:
        originals = {r["material_id"]: r for r in csv.DictReader(stream)}
    assets = []
    for cls in CLASSES:
        train = sorted(
            (r for r in rows if r["class"] == cls and r["split"] == "train"),
            key=lambda r: r["material_id"],
        )
        if not train:
            raise ValueError(f"no old development source for {cls}")
        row = train[0]
        original = originals.get(row["material_id"])
        if original is None or any(original[k] != row[k] for k in original):
            raise ValueError("development split manifest does not match original source record")
        maps = {}
        for role, key in (
            ("color", "base_color_file"),
            ("roughness", "roughness_file"),
            ("normal", "normal_file"),
            ("metallic", "metallic_file"),
        ):
            if row[key]:
                path = inside(
                    args.development_materials_root, f"{cls}/{row['material_id']}/{row[key]}"
                )
                if not path.is_file():
                    raise ValueError(f"missing old development map: {path}")
                maps[role] = {
                    "path": str(path),
                    "sha256": digest(path),
                    "colorspace": "sRGB" if role == "color" else "Non-Color",
                }
        if row["normal_variant"] != "gl" or not all(
            k in maps for k in ("color", "roughness", "normal")
        ):
            raise ValueError("old development source must have complete GL PBR maps")
        assets.append(
            {
                "class": cls,
                "material_id": row["material_id"],
                "asset_id": row["source_material_name"],
                "provider": "old_development",
                "split": "development",
                "maps": maps,
                "metallic_value": float(row["metallic_value"]),
            }
        )
    return assets[:1] if args.mode == "smoke" else assets


def build_scene(config, smoke):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES" if smoke else config["engine"]
    if smoke:
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
    else:
        for key, value in config["eevee"].items():
            setattr(scene.eevee, key, value)
        for key, value in config["eevee_raytracing"].items():
            setattr(scene.eevee.ray_tracing_options, key, value)
    scene.render.resolution_x, scene.render.resolution_y = (
        [64, 64] if smoke else config["resolution"]
    )
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.display_settings.display_device = "sRGB"
    for key, value in config["color_management"].items():
        setattr(scene.view_settings, key, value)
    world = bpy.data.worlds.new("HoldoutWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = config["world_color"]
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = config["world_strength"]
    scene.world = world
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96, ring_count=48, radius=1, location=config["sphere_center"]
    )
    sphere = bpy.context.object
    sphere.name = "HoldoutSphere"
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    bpy.ops.mesh.primitive_plane_add(size=config["floor_size"], location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "HoldoutGround"
    ground = bpy.data.materials.new("HoldoutGroundMaterial")
    ground.use_nodes = True
    bsdf = ground.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = config["floor_color"]
    bsdf.inputs["Roughness"].default_value = config["floor_roughness"]
    floor.data.materials.append(ground)
    bpy.ops.object.camera_add(location=(0, -4.1599531173706055, 2.9398179054260254))
    camera = bpy.context.object
    camera.name = "HoldoutCamera"
    camera.data.lens = config["camera_lens_mm"]
    camera.data.sensor_width = config["camera_sensor_width_mm"]
    camera.data.sensor_fit = "AUTO"
    camera.data.clip_start, camera.data.clip_end = 0.1, 1000
    camera.data.dof.use_dof = False
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(5.5, -1.5, 4.4))
    light = bpy.context.object
    light.name = "HoldoutArea"
    light.data.energy = 2000
    light.data.shape = config["light_shape"]
    light.data.size = config["light_size"]
    light.data.color = (1, 1, 1)
    return scene, sphere, camera, light


def material(asset, root):
    import bpy

    mat = bpy.data.materials.new("Holdout_" + asset["material_id"])
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bsdf.inputs["Metallic"].default_value = asset["metallic_value"]
    uv, mapping = nodes.new("ShaderNodeTexCoord"), nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1, 1, 1)
    links.new(uv.outputs["UV"], mapping.inputs["Vector"])
    for role, spec in asset["maps"].items():
        path = inside(root, spec["path"]) if root else Path(spec["path"])
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(path), check_existing=False)
        texture.image.colorspace_settings.name = spec["colorspace"]
        texture.interpolation, texture.extension = "Linear", "REPEAT"
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        if role == "normal":
            normal = nodes.new("ShaderNodeNormalMap")
            normal.space = "TANGENT"
            normal.inputs["Strength"].default_value = 1.0
            links.new(texture.outputs["Color"], normal.inputs["Color"])
            links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            socket = {"color": "Base Color", "roughness": "Roughness", "metallic": "Metallic"}[role]
            links.new(texture.outputs["Color"], bsdf.inputs[socket])
    return mat


def scene_fingerprint(scene, sphere, config):
    """Fingerprint the realized static scene as well as its frozen specification."""
    world = scene.world.node_tree.nodes["Background"]
    # Blender's primitive builder can emit polygon/loop records in a different
    # storage order between processes. Preserve topology, winding and vertex-UV
    # associations, but canonicalize face order and cyclic corner starting points.
    mesh = sphere.data
    uv_layer = mesh.uv_layers[0].data
    faces = []
    for polygon in mesh.polygons:
        corners = tuple(
            (*mesh.vertices[mesh.loops[i].vertex_index].co, *uv_layer[i].uv)
            for i in polygon.loop_indices
        )
        faces.append(min(corners[i:] + corners[:i] for i in range(len(corners))))
    payload = {
        "config_scene": config,
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "vertices": sorted(tuple(v.co) for v in sphere.data.vertices),
        "faces_with_uv": sorted(faces),
        "world_color": tuple(world.inputs["Color"].default_value),
        "world_strength": world.inputs["Strength"].default_value,
        "lens": scene.camera.data.lens,
        "sensor_width": scene.camera.data.sensor_width,
        "floor_size": list(scene.objects["HoldoutGround"].dimensions),
        "light_shape": scene.objects["HoldoutArea"].data.shape,
        "light_size": scene.objects["HoldoutArea"].data.size,
    }
    return fingerprint(payload)


def existing_rows(manifest, out, planned, width, height):
    rows = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("stale/unsupported manifest columns")
            for row in reader:
                relative = row["image_path"]
                if relative in rows or relative not in planned:
                    raise ValueError("duplicate or unexpected manifest image")
                expected = planned[relative]
                # Compare all deterministic metadata, not only a top-level run hash.
                if any(row[k] != str(v) for k, v in expected.items()):
                    raise ValueError(f"stale render metadata: {relative}")
                if validate_png(inside(out, relative), width, height) != row["image_sha256"]:
                    raise ValueError(f"stale/corrupt PNG: {relative}")
                rows[relative] = row
    disk = {p.relative_to(out).as_posix() for p in out.rglob("*.png")} if out.exists() else set()
    if disk != set(rows):
        raise ValueError(
            "orphan/missing PNGs; quarantine the run instead of silently reusing or overwriting it"
        )
    return rows


def main(argv=None):
    args = parse_args(argv)
    validate_paths(args)
    config = read_json(args.config)
    validate_config(config)
    if digest(args.original_manifest) != config["original_manifest_sha256"]:
        raise ValueError("original source manifest changed")
    import bpy
    from mathutils import Vector

    if tuple(bpy.app.version) != tuple(config["blender_version"]):
        raise ValueError(
            f"requires Blender {config['blender_version']}; got {bpy.app.version_string}"
        )
    if bpy.data.filepath:
        raise ValueError("do not open any .blend input; use --factory-startup")
    if args.mode == "holdout":
        registry, receipt = read_json(args.registry), read_json(args.receipt)
        validate_registry(registry, args.original_manifest)
        validate_identity_audit(registry, read_json(args.identity_audit))
        if digest(args.identity_audit) != config["identity_audit_sha256"]:
            raise ValueError("source identity audit changed since protocol freeze")
        validate_receipt(registry, receipt, args.sources_root)
        if config["selection_sha256"] != registry["selection_sha256"]:
            raise ValueError("configuration does not match the frozen selection")
        assets, source_hash = registry["assets"], fingerprint(receipt)
        selection_hash = registry["selection_sha256"]
    else:
        assets = development_assets(args)
        source_hash = fingerprint(assets)
        selection_hash = fingerprint({"mode": args.mode, "assets": assets})
    smoke = args.mode == "smoke"
    scene, sphere, camera, light = build_scene(config, smoke)
    scene_hash = scene_fingerprint(scene, sphere, config)
    renderer_hash = fingerprint({p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})
    runtime = {
        "blender_version": bpy.app.version_string,
        "build_hash": bpy.app.build_hash.decode(),
        "executable_sha256": digest(bpy.app.binary_path),
        "engine": scene.render.engine,
        "mode": args.mode,
        "device": "CPU" if smoke else "EEVEE_GPU",
    }
    if not smoke:
        import gpu

        runtime["gpu"] = {
            "vendor": gpu.platform.vendor_get(),
            "renderer": gpu.platform.renderer_get(),
            "version": gpu.platform.version_get(),
            "backend": gpu.platform.backend_type_get(),
        }
    run = {
        "selection_sha256": selection_hash,
        "source_sha256": source_hash,
        "renderer_sha256": renderer_hash,
        "config_sha256": digest(args.config),
        "scene_sha256": scene_hash,
        "runtime": runtime,
    }
    run_hash = fingerprint(run)
    width, height = scene.render.resolution_x, scene.render.resolution_y
    planned, jobs = {}, []
    for asset in assets:
        for index in range(1, (30 if args.mode == "holdout" else 1) + 1):
            state = parameters(config, asset["class"], asset["material_id"], index)
            filename = f"{asset['material_id']}_r{index:02d}.png"
            relative = f"{asset['class']}/{filename}"
            row = {
                "image_path": relative,
                "filename": filename,
                "class": asset["class"],
                "material_id": asset["material_id"],
                "source_asset_id": asset["asset_id"],
                "provider": asset["provider"],
                "split": "holdout" if args.mode == "holdout" else args.mode,
                "original_split": "holdout" if args.mode == "holdout" else "train",
                "render_index": index,
                **state,
                "render_engine": scene.render.engine,
                "blender_version": bpy.app.version_string,
                "resolution_width": width,
                "resolution_height": height,
                **{k: v for k, v in run.items() if k != "runtime"},
                "run_sha256": run_hash,
                "parameters_sha256": fingerprint(state),
            }
            planned[relative] = row
            jobs.append((asset, state, row))
    # Validate every existing output before any render or manifest modification.
    rows = existing_rows(args.manifest, args.out, planned, width, height)
    run_path = args.out / "run_fingerprint.json"
    if run_path.exists() and read_json(run_path) != {**run, "run_sha256": run_hash}:
        raise ValueError("stale scene/source/renderer/config/runtime fingerprints")
    if rows and not run_path.exists():
        raise ValueError("missing run fingerprint")
    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(run_path, {**run, "run_sha256": run_hash})
    fresh = not args.manifest.exists()
    previous, mat = None, None
    with args.manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for asset, state, row in jobs:
            if row["image_path"] in rows:
                continue
            if previous != asset["material_id"]:
                sphere.data.materials.clear()
                if mat:
                    images = [
                        node.image for node in mat.node_tree.nodes if node.type == "TEX_IMAGE"
                    ]
                    bpy.data.materials.remove(mat)
                    for image in images:
                        bpy.data.images.remove(image)
                mat = material(asset, args.sources_root if args.mode == "holdout" else None)
                sphere.data.materials.append(mat)
                previous = asset["material_id"]
            sphere.rotation_euler = [math.radians(x) for x in state["object_rotation_deg"]]
            camera.location = state["camera_location"]
            light.location, light.data.energy = state["light_location"], state["light_power"]
            for obj in (camera, light):
                obj.rotation_euler = (
                    (Vector(config["sphere_center"]) - obj.location)
                    .to_track_quat("-Z", "Y")
                    .to_euler()
                )
            if smoke:
                scene.cycles.seed = state["random_seed"]
            target = inside(args.out, row["image_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(target)
            bpy.ops.render.render(write_still=True)
            row["image_sha256"] = validate_png(target, width, height)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            print(f"HOLDOUT_RENDER {row['image_path']}", flush=True)
    print(f"COMPLETE mode={args.mode} expected={len(planned)} resumed={len(rows)}")


if __name__ == "__main__":
    if "--" not in sys.argv:
        raise SystemExit("pass explicit renderer arguments after Blender's -- separator")
    main(sys.argv[sys.argv.index("--") + 1 :])

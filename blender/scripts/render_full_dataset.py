"""Full dataset render - resumable, deterministic, headless-capable.

    blender --background blender/material_dataset_final_render.blend \
            --python blender/scripts/render_full_dataset.py -- [options]

Options (after --):
    --renders N        renders per base material (default 30)
    --materials a,b    restrict to these material_ids (smoke tests)
    --out DIR          output root (default <project>/dataset)
    --manifest PATH    manifest csv (default <project>/metadata/render_manifest.csv)
    --no-save          do not save the .blend at the end

Randomization logic is imported verbatim from generate_dataset.py - the same
Baseline / CameraOrbit / child_seed / RenderSynth pipeline used by the
validated pilot. Nothing is redefined here.

Resumability: an image already on disk that is a readable 512x512 PNG is not
re-rendered. Because every child seed is derived from
(master, class, material_id, render_index) rather than from RNG call order,
skipping renders cannot change any other render's parameters.
"""

import csv
import os
import random
import struct
import sys
import time

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for _path in (SCRIPT_DIR, os.path.join(PROJECT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import generate_dataset as gd        # Baseline, CameraOrbit, look_at, child_seed
import configure_rendersynth as cr

REPORT_DIR = os.path.join(PROJECT, "reports", "dataset_generation")

RENDERS_PER_MATERIAL = 30
EXPECTED_SIZE = (512, 512)

FIELDS = [
    "image_path", "filename", "class", "material_id", "blender_material_name",
    "split", "render_index", "random_seed",
    "camera_location_x", "camera_location_y", "camera_location_z",
    "camera_target_x", "camera_target_y", "camera_target_z",
    "camera_rotation_x", "camera_rotation_y", "camera_rotation_z",
    "camera_azimuth_deg", "camera_elevation_deg", "camera_distance",
    "area_light_location_x", "area_light_location_y", "area_light_location_z",
    "area_light_power",
    "object_rotation_x", "object_rotation_y", "object_rotation_z",
    "render_engine", "resolution_width", "resolution_height",
    "source_base_material", "material_verified_on_object",
]


# --------------------------------------------------------------------- helpers

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {"renders": RENDERS_PER_MATERIAL, "materials": None,
            "out": os.path.join(PROJECT, "dataset"),
            "manifest": os.path.join(PROJECT, "metadata", "render_manifest.csv"),
            "save": True}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--renders":
            opts["renders"] = int(argv[i + 1]); i += 2
        elif a == "--materials":
            opts["materials"] = set(argv[i + 1].split(",")); i += 2
        elif a == "--out":
            opts["out"] = argv[i + 1]; i += 2
        elif a == "--manifest":
            opts["manifest"] = argv[i + 1]; i += 2
        elif a == "--no-save":
            opts["save"] = False; i += 1
        else:
            i += 1
    return opts


def png_size(path):
    """Read a PNG's dimensions from its IHDR without decoding pixels."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except Exception:
        return None


def image_ok(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    return png_size(path) == EXPECTED_SIZE


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------- render

def main():
    opts = parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    scene = bpy.context.scene
    obj = bpy.data.objects[gd.TARGET_OBJECT]
    light = bpy.data.objects[gd.LIGHT_OBJECT]
    cam = bpy.data.objects[gd.CAMERA_OBJECT]

    rows = gd.load_split_manifest()
    if opts["materials"]:
        rows = [r for r in rows if r["_id"] in opts["materials"]]

    # ---- preflight: every manifest row must resolve to a usable material
    problems = []
    for rec in rows:
        name = rec["blender_material_name"]
        mat = bpy.data.materials.get(name)
        if mat is None:
            problems.append([rec["_id"], "missing material '%s'" % name])
            continue
        if mat.get("base_material_id") != rec["_id"]:
            problems.append([rec["_id"], "material '%s' has base_material_id=%r"
                             % (name, mat.get("base_material_id"))])
        if rec["class"] not in cr_classes():
            problems.append([rec["_id"], "bad class %r" % rec["class"]])
        if rec["split"] not in ("train", "val", "test"):
            problems.append([rec["_id"], "bad split %r" % rec["split"]])
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE":
                if n.image is None or not os.path.isfile(bpy.path.abspath(n.image.filepath)):
                    problems.append([rec["_id"], "unresolved texture on %s" % n.name])
        out_dir = os.path.join(opts["out"], rec["split"], rec["class"])
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            problems.append([rec["_id"], "cannot create %s: %s" % (out_dir, exc)])

    if problems:
        log("PREFLIGHT FAILED (%d problems):" % len(problems))
        for p in problems:
            log("   %s : %s" % tuple(p))
        print("PREFLIGHT_FAILED", flush=True)
        return
    log("preflight OK - %d materials, %d renders each = %d images"
        % (len(rows), opts["renders"], len(rows) * opts["renders"]))

    # ---- existing manifest rows (resume)
    existing = {}
    if os.path.isfile(opts["manifest"]):
        try:
            with open(opts["manifest"], newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    existing[(r["material_id"], int(r["render_index"]))] = r
        except Exception as exc:
            log("could not read existing manifest (%s); starting fresh" % exc)
    log("manifest carries %d existing rows" % len(existing))

    baseline = gd.Baseline()
    orbit = gd.CameraOrbit(baseline)
    pipeline = gd.ExecutablePipeline(bpy.context, scene.pipeline_data)
    full_ctx = pipeline.build_context_manager()

    width, height = scene.render.resolution_x, scene.render.resolution_y
    engine = scene.render.engine

    out_rows = []
    rendered = skipped = failed = 0
    failures = []
    t_start = time.time()
    total = len(rows) * opts["renders"]

    # incremental manifest: append as we go, flush every row
    fresh = not os.path.isfile(opts["manifest"])
    mf = open(opts["manifest"], "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()
        mf.flush()

    try:
        with full_ctx:
            for mi, rec in enumerate(rows, start=1):
                cls, mid = rec["class"], rec["_id"]
                mat = bpy.data.materials[rec["blender_material_name"]]
                out_dir = os.path.join(opts["out"], rec["split"], cls)
                m_rendered = m_skipped = 0

                for idx in range(1, opts["renders"] + 1):
                    filename = "%s_r%02d.png" % (mid, idx)
                    out_path = os.path.join(out_dir, filename)
                    key = (mid, idx)

                    if image_ok(out_path) and key in existing:
                        out_rows.append(existing[key])
                        skipped += 1
                        m_skipped += 1
                        continue

                    seed = gd.child_seed(cls, mid, idx)
                    random.seed(seed)
                    try:
                        import numpy as np
                        np.random.seed(seed % (2 ** 32))
                    except ImportError:
                        pass

                    obj.data.materials[0] = mat
                    try:
                        with full_ctx.frame_context():
                            pipeline.execute()
                            orbit_info = orbit.apply()
                            light.rotation_euler = gd.look_at(light.location,
                                                              baseline.target)
                            scene.render.filepath = out_path
                            bpy.ops.render.render(write_still=True)

                            row = {
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
                                "camera_target_x": round(baseline.target.x, 6),
                                "camera_target_y": round(baseline.target.y, 6),
                                "camera_target_z": round(baseline.target.z, 6),
                                "camera_rotation_x": round(_deg(cam.rotation_euler.x), 4),
                                "camera_rotation_y": round(_deg(cam.rotation_euler.y), 4),
                                "camera_rotation_z": round(_deg(cam.rotation_euler.z), 4),
                                "camera_azimuth_deg": round(orbit_info["azimuth_deg"], 4),
                                "camera_elevation_deg": round(orbit_info["elevation_deg"], 4),
                                "camera_distance": round(orbit_info["distance"], 6),
                                "area_light_location_x": round(light.location.x, 6),
                                "area_light_location_y": round(light.location.y, 6),
                                "area_light_location_z": round(light.location.z, 6),
                                "area_light_power": round(light.data.energy, 3),
                                "object_rotation_x": round(_deg(obj.rotation_euler.x), 4),
                                "object_rotation_y": round(_deg(obj.rotation_euler.y), 4),
                                "object_rotation_z": round(_deg(obj.rotation_euler.z), 4),
                                "render_engine": engine,
                                "resolution_width": width,
                                "resolution_height": height,
                                "source_base_material": rec.get("source_folder", ""),
                                "material_verified_on_object": obj.data.materials[0].name,
                            }
                        if not image_ok(out_path):
                            raise RuntimeError("output missing or wrong size")
                        writer.writerow(row)
                        mf.flush()
                        os.fsync(mf.fileno())
                        out_rows.append(row)
                        rendered += 1
                        m_rendered += 1
                    except Exception as exc:
                        failed += 1
                        failures.append([filename, "%s: %s" % (type(exc).__name__, exc)])
                        log("FAIL %s : %s" % (filename, exc))

                done = rendered + skipped + failed
                el = time.time() - t_start
                rate = rendered / el if (rendered and el) else 0
                eta = ((total - done) / rate / 60.0) if rate else 0
                log("[%3d/%3d] %-12s %-8s %-5s  rendered=%2d skipped=%2d  "
                    "total %4d/%4d  %.2fs/img  ETA %.0f min"
                    % (mi, len(rows), mid, cls, rec["split"], m_rendered, m_skipped,
                       done, total, (el / rendered if rendered else 0), eta))
    finally:
        mf.close()
        baseline.restore()

    elapsed = time.time() - t_start

    # ---- canonical manifest: one row per (material, index), ordered
    canon = {}
    for r in out_rows:
        canon[(r["material_id"], int(r["render_index"]))] = r
    ordered = [canon[k] for k in sorted(canon, key=lambda k: (k[0], k[1]))]
    with open(opts["manifest"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)

    if opts["save"]:
        bpy.ops.wm.save_mainfile()

    log("DONE rendered=%d skipped=%d failed=%d manifest_rows=%d elapsed=%.1fs"
        % (rendered, skipped, failed, len(ordered), elapsed))
    if failures:
        log("FAILURES:")
        for f in failures:
            log("   %s : %s" % tuple(f))
    print("RENDER_SUMMARY:%s" % {
        "rendered": rendered, "skipped": skipped, "failed": failed,
        "manifest_rows": len(ordered), "elapsed_s": round(elapsed, 1),
        "failures": failures}, flush=True)


def _deg(rad):
    import math
    return math.degrees(rad)


def cr_classes():
    import material_scan
    return material_scan.CLASSES


if __name__ == "__main__":
    main()

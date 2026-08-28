"""Baseline audit render: one image per base material, zero randomization.

Blender side  : render_classes() / probe_images()
Analysis side : analyse() / contact_sheet() / write_report()  (needs only PIL)

Every render uses the validated baseline camera, light and sphere transform.
The RenderSynth pipeline is NOT executed here.
"""

import csv
import json
import math
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT_DIR = os.path.join(PROJECT, "material_audit")
SPLIT_MANIFEST = os.path.join(PROJECT, "metadata", "material_manifest_split.csv")
PROBE_JSON = os.path.join(PROJECT, "material_audit", "_texture_probe.json")
REPORT_MD = os.path.join(PROJECT, "reports", "dataset_generation", "material_audit_report.md")
REPORT_CSV = os.path.join(PROJECT, "reports", "dataset_generation", "material_audit_report.csv")
CONTACT_SHEET = os.path.join(PROJECT, "reports", "dataset_generation", "material_audit_contact_sheet.png")

CLASS_ORDER = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]

# Baseline geometry: camera aims at the sphere centre, so the silhouette is a
# centred circle. Sample well inside it to measure the material only.
LENS_MM, SENSOR_MM, SPHERE_R = 50.0, 36.0, 1.0
DISC_FRACTION = 0.80          # of the silhouette radius

# Flag thresholds
UNIFORM_STD = 4.0             # luminance stddev inside the disc
DARK_MEAN, BRIGHT_MEAN = 18.0, 238.0
DETAIL_MIN = 1.2              # mean |pixel - local mean|, isolates texture from shading


def load_materials():
    with open(SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_id"] = r.get("base_material_id") or r["material_id"]
    rows.sort(key=lambda r: (CLASS_ORDER.index(r["class"]), r["_id"]))
    return rows


# =========================================================== Blender side

def render_classes(classes):
    """Render every material of the named classes at the untouched baseline."""
    import bpy
    import sys
    sd = os.path.join(PROJECT, "blender", "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import generate_dataset as gd

    scene = bpy.context.scene
    obj = bpy.data.objects["material_object"]
    baseline = gd.Baseline()          # captures camera, light, rotation, material

    done = []
    try:
        for rec in load_materials():
            if rec["class"] not in classes:
                continue
            out_dir = os.path.join(AUDIT_DIR, rec["class"])
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, rec["_id"] + ".png")

            obj.data.materials[0] = bpy.data.materials[rec["blender_material_name"]]
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            done.append(rec["_id"])
    finally:
        baseline.restore()            # nothing was randomized; this just re-seats the material
    return done


def probe_images():
    """Record, per material, whether each texture resolved and actually loaded."""
    import bpy
    out = {}
    for rec in load_materials():
        mat = bpy.data.materials[rec["blender_material_name"]]
        entries = []
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.name.startswith("TEX_"):
                continue
            img = node.image
            if img is None:
                entries.append({"slot": node.name[4:], "image": None,
                                "exists": False, "loaded": False})
                continue
            abspath = bpy.path.abspath(img.filepath)
            # Blender lazy-loads images, so has_data only means "currently in
            # memory" - it is False for every texture after a fresh file open.
            # Force an actual decode so this tests real loadability.
            if not img.has_data:
                try:
                    img.reload()
                except Exception:
                    pass
            entries.append({
                "slot": node.name[4:],
                "image": img.name,
                "filepath": img.filepath,
                "exists": os.path.isfile(abspath),
                "loaded": tuple(img.size) != (0, 0),
                "size": list(img.size),
                "colorspace": img.colorspace_settings.name,
            })
        out[rec["_id"]] = entries
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(PROBE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return {"materials_probed": len(out)}


# =========================================================== analysis side

def _disc_stats(path):
    """Luminance statistics inside the sphere silhouette only."""
    from PIL import Image, ImageFilter
    with Image.open(path) as im:
        g = im.convert("L")
        w, h = g.size
    half_fov = math.atan((SENSOR_MM / 2) / LENS_MM)
    f_px = (w / 2) / math.tan(half_fov)
    # baseline distance recovered from the scene geometry used for the audit
    d = 4.590000386913341
    r_px = f_px * math.tan(math.asin(SPHERE_R / d)) * DISC_FRACTION
    cx, cy = w / 2.0, h / 2.0

    with Image.open(path) as im:
        g = im.convert("L")
        blur = g.filter(ImageFilter.BoxBlur(3))
        gp, bp = g.load(), blur.load()
        vals, detail = [], []
        r2 = r_px * r_px
        y0, y1 = int(cy - r_px), int(cy + r_px) + 1
        x0, x1 = int(cx - r_px), int(cx + r_px) + 1
        for y in range(max(0, y0), min(h, y1)):
            dy2 = (y - cy) ** 2
            for x in range(max(0, x0), min(w, x1)):
                if (x - cx) ** 2 + dy2 <= r2:
                    v = gp[x, y]
                    vals.append(v)
                    detail.append(abs(v - bp[x, y]))
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "n_px": n,
        "mean": round(mean, 2),
        "std": round(math.sqrt(var), 2),
        "min": min(vals),
        "max": max(vals),
        "detail": round(sum(detail) / n, 3),
        # brightness-normalised: absolute pixel differences shrink on dark
        # materials, so this is the fairer "is there texture here" measure
        "detail_rel": round(100.0 * (sum(detail) / n) / max(mean, 1.0), 3),
    }


def analyse():
    probe = {}
    if os.path.isfile(PROBE_JSON):
        with open(PROBE_JSON, encoding="utf-8") as fh:
            probe = json.load(fh)

    rows = []
    for rec in load_materials():
        path = os.path.join(AUDIT_DIR, rec["class"], rec["_id"] + ".png")
        row = {"class": rec["class"], "material_id": rec["_id"],
               "blender_material_name": rec["blender_material_name"],
               "split": rec["split"],
               "image": os.path.relpath(path, PROJECT).replace("\\", "/")}

        tex = probe.get(rec["_id"], [])
        bad_tex = [t["slot"] for t in tex if not t.get("exists") or not t.get("loaded")]
        row["textures_expected"] = len(tex)
        row["textures_failed"] = ",".join(bad_tex)

        if not os.path.isfile(path):
            row.update({"status": "MISSING_RENDER", "flags": "missing_render"})
            rows.append(row)
            continue

        st = _disc_stats(path)
        row.update(st)

        flags = []
        if bad_tex:
            flags.append("texture_load_failed")
        if st["std"] < UNIFORM_STD:
            flags.append("almost_uniform")
        if st["mean"] < DARK_MEAN:
            flags.append("nearly_black")
        if st["mean"] > BRIGHT_MEAN:
            flags.append("nearly_white")
        if st["detail"] < DETAIL_MIN:
            flags.append("no_visible_texture")
        row["flags"] = ",".join(flags)
        row["status"] = "FLAGGED" if flags else "ok"
        rows.append(row)
    return rows


def contact_sheet(rows, tile=160, label_h=17, header_h=24, pad=3):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
        head = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = head = ImageFont.load_default()

    per_row = 15
    cell_w = tile + pad
    cell_h = tile + label_h + pad
    row_h = header_h + cell_h
    W = per_row * cell_w + pad
    H = len(CLASS_ORDER) * row_h + pad

    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)

    by_class = {c: [r for r in rows if r["class"] == c] for c in CLASS_ORDER}
    for ci, cls in enumerate(CLASS_ORDER):
        top = pad + ci * row_h
        items = by_class[cls]
        d.rectangle([0, top, W, top + header_h - 2], fill=(38, 38, 44))
        d.text((6, top + 4), "%s  -  %d materials" % (cls.upper(), len(items)),
               fill=(245, 245, 245), font=head)
        for i, r in enumerate(items[:per_row]):
            x = pad + i * cell_w
            y = top + header_h
            p = os.path.join(PROJECT, r["image"])
            if os.path.isfile(p):
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((tile, tile)), (x, y))
            else:
                d.rectangle([x, y, x + tile, y + tile], fill=(60, 20, 20))
            flagged = bool(r.get("flags"))
            if flagged:
                d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(235, 140, 40), width=3)
            d.rectangle([x, y + tile, x + tile, y + tile + label_h], fill=(28, 28, 32))
            d.text((x + 4, y + tile + 3), r["material_id"],
                   fill=((250, 180, 90) if flagged else (215, 215, 220)), font=font)
    sheet.save(CONTACT_SHEET)
    return CONTACT_SHEET


def write_report(rows):
    fields = ["class", "material_id", "blender_material_name", "split", "image",
              "textures_expected", "textures_failed", "n_px", "mean", "std",
              "min", "max", "detail", "detail_rel", "status", "flags"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    flagged = [r for r in rows if r.get("flags")]
    by_flag = {}
    for r in flagged:
        for f in r["flags"].split(","):
            by_flag.setdefault(f, []).append(r)

    lines = []
    lines.append("# Base-material audit\n")
    lines.append("One 512x512 baseline render per base material. No randomization: "
                 "validated camera, light position/power, sphere transform, texture "
                 "mapping and material parameters all untouched.\n")
    lines.append("- Materials audited: **%d**" % len(rows))
    lines.append("- Clean: **%d**" % (len(rows) - len(flagged)))
    lines.append("- Flagged: **%d**\n" % len(flagged))
    lines.append("Measurements are taken inside the sphere silhouette only "
                 "(central disc at %d%% of the projected radius), so the neutral "
                 "background never dilutes them. `detail` is mean |pixel - local mean|, "
                 "which separates real texture from smooth shading.\n"
                 % int(DISC_FRACTION * 100))
    lines.append("## Thresholds\n")
    lines.append("| Flag | Condition |")
    lines.append("|---|---|")
    lines.append("| `almost_uniform` | luminance std < %.1f |" % UNIFORM_STD)
    lines.append("| `nearly_black` | mean < %.0f |" % DARK_MEAN)
    lines.append("| `nearly_white` | mean > %.0f |" % BRIGHT_MEAN)
    lines.append("| `no_visible_texture` | detail < %.1f |" % DETAIL_MIN)
    lines.append("| `texture_load_failed` | a map file is missing or failed to load |")
    lines.append("")

    if not flagged:
        lines.append("## Result\n\nNo material tripped any flag.\n")
    else:
        lines.append("## Flagged materials\n")
        for flag, items in sorted(by_flag.items()):
            lines.append("### `%s` - %d material(s)\n" % (flag, len(items)))
            lines.append("| material_id | class | split | mean | std | detail | detail_rel | all flags |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in sorted(items, key=lambda r: (r["class"], r["material_id"])):
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["material_id"], r["class"], r["split"], r.get("mean", "-"),
                    r.get("std", "-"), r.get("detail", "-"),
                    r.get("detail_rel", "-"), r["flags"]))
            lines.append("")

    lines.append("## Lowest-detail materials (texture visibility ranking)\n")
    lines.append("`detail` is absolute, so it shrinks on dark materials; "
                 "`detail_rel` normalises by mean luminance and is the fairer "
                 "cross-class comparison. Both rankings are given.")
    lines.append("")
    lines.append("### By absolute detail")
    lines.append("")
    ranked = sorted((r for r in rows if "detail" in r), key=lambda r: r["detail"])[:12]
    lines.append("| material_id | class | detail | detail_rel | std | mean |")
    lines.append("|---|---|---|---|---|---|")
    for r in ranked:
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (r["material_id"], r["class"], r["detail"],
                        r["detail_rel"], r["std"], r["mean"]))
    lines.append("")
    lines.append("### By brightness-normalised detail")
    lines.append("")
    ranked2 = sorted((r for r in rows if "detail_rel" in r),
                     key=lambda r: r["detail_rel"])[:12]
    lines.append("| material_id | class | detail_rel | detail | std | mean |")
    lines.append("|---|---|---|---|---|---|")
    for r in ranked2:
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (r["material_id"], r["class"], r["detail_rel"],
                        r["detail"], r["std"], r["mean"]))
    lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return {"md": REPORT_MD, "csv": REPORT_CSV, "flagged": len(flagged)}


if __name__ == "__main__":
    rows = analyse()
    sheet = contact_sheet(rows)
    rep = write_report(rows)
    print("audited      : %d" % len(rows))
    print("flagged      : %d" % rep["flagged"])
    print("contact sheet: %s" % sheet)
    print("report       : %s" % rep["md"])
    for r in rows:
        if r.get("flags"):
            print("  FLAG %-12s %-9s %s" % (r["material_id"], r["class"], r["flags"]))

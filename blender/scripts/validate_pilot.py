"""Validate the pilot render batch and build the contact sheet.

Runs outside Blender - it only reads the PNGs and render_manifest.csv.
Framing is checked analytically from the recorded camera geometry (exact),
then cross-checked against the pixels for ground and contact shadow.
"""

import csv
import math
import os
import statistics
from collections import defaultdict

from PIL import Image, ImageDraw

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(PROJECT, "pilot_dataset")
RENDER_MANIFEST = os.path.join(PROJECT, "metadata", "render_manifest.csv")
SPLIT_MANIFEST = os.path.join(PROJECT, "metadata", "material_manifest_split.csv")
CONTACT_SHEET = os.path.join(PROJECT, "reports", "dataset_generation", "pilot_contact_sheet.png")

CLASS_ORDER = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]

EXPECTED_TOTAL = 35
EXPECTED_PER_CLASS = 5
EXPECTED_SIZE = (512, 512)

SPHERE_RADIUS = 1.0
LENS_MM = 50.0
SENSOR_MM = 36.0

# Sphere height as a fraction of frame height must stay in this band.
FRAME_FRACTION = (0.55, 0.75)


def _rows():
    with open(RENDER_MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _split_lookup():
    with open(SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        out = {}
        for r in csv.DictReader(fh):
            out[r.get("base_material_id") or r["material_id"]] = r
        return out


def check_files(rows):
    res = {"png_count": 0, "per_class": {}, "collisions": [], "wrong_size": [],
           "zero_byte": [], "corrupt": [], "missing": []}
    seen = defaultdict(list)
    for root, _d, files in os.walk(PILOT):
        for f in files:
            if f.lower().endswith(".png"):
                res["png_count"] += 1
                cls = os.path.basename(root)
                res["per_class"][cls] = res["per_class"].get(cls, 0) + 1
                seen[f].append(os.path.join(root, f))
    res["collisions"] = [f for f, paths in seen.items() if len(paths) > 1]

    for r in rows:
        p = os.path.join(PROJECT, r["image_path"])
        if not os.path.isfile(p):
            res["missing"].append(r["image_path"])
            continue
        if os.path.getsize(p) == 0:
            res["zero_byte"].append(r["image_path"])
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.size != EXPECTED_SIZE:
                    res["wrong_size"].append([r["image_path"], im.size])
        except Exception as exc:
            res["corrupt"].append([r["image_path"], str(exc)])
    return res


def check_materials(rows, split_map):
    res = {"material_mismatch": [], "class_mismatch": [], "split_mismatch": [],
           "not_in_split_manifest": []}
    for r in rows:
        if r["material_verified_on_object"] != r["blender_material_name"]:
            res["material_mismatch"].append(
                [r["filename"], r["blender_material_name"], r["material_verified_on_object"]])
        src = split_map.get(r["material_id"])
        if src is None:
            res["not_in_split_manifest"].append(r["material_id"])
            continue
        if src["class"] != r["class"]:
            res["class_mismatch"].append([r["filename"], src["class"], r["class"]])
        if src["split"] != r["split"]:
            res["split_mismatch"].append([r["filename"], src["split"], r["split"]])
    return res


def check_framing(rows):
    """Exact: the camera always aims at the sphere centre, so the silhouette is a
    circle centred in frame whose pixel radius follows from the recorded distance."""
    res = {"cropped": [], "too_small": [], "too_large": [], "no_ground_below": [],
           "fraction_range": None, "no_contact_shadow": []}
    half_fov = math.atan((SENSOR_MM / 2) / LENS_MM)
    fractions = []

    for r in rows:
        w, h = EXPECTED_SIZE
        f_px = (w / 2) / math.tan(half_fov)
        d = float(r["camera_distance"])
        theta = math.asin(SPHERE_RADIUS / d)
        r_px = f_px * math.tan(theta)
        frac = (2 * r_px) / h
        fractions.append(frac)

        cx, cy = w / 2.0, h / 2.0
        if cx - r_px < 0 or cx + r_px > w or cy - r_px < 0 or cy + r_px > h:
            res["cropped"].append([r["filename"], round(r_px, 1)])
        if frac < FRAME_FRACTION[0]:
            res["too_small"].append([r["filename"], round(frac, 3)])
        if frac > FRAME_FRACTION[1]:
            res["too_large"].append([r["filename"], round(frac, 3)])

        bottom = cy + r_px
        if bottom >= h - 2:
            res["no_ground_below"].append([r["filename"], round(bottom, 1)])

        # contact shadow: the darkest ground band under the sphere should be
        # clearly darker than the far ground at the bottom corners
        p = os.path.join(PROJECT, r["image_path"])
        if os.path.isfile(p):
            with Image.open(p) as im:
                g = im.convert("L")
                y0 = int(min(h - 1, bottom))
                band = g.crop((int(cx - r_px), y0, int(cx + r_px), min(h, y0 + 30)))
                corner = g.crop((0, h - 30, 40, h))
                if band.size[0] > 0 and band.size[1] > 0:
                    band_min = min(band.getdata())
                    corner_med = statistics.median(corner.getdata())
                    if corner_med - band_min < 10:
                        res["no_contact_shadow"].append(
                            [r["filename"], band_min, corner_med])

    res["fraction_range"] = [round(min(fractions), 3), round(max(fractions), 3)]
    return res


def check_exposure(rows):
    res = {"too_dark": [], "too_bright": [], "clipped_white": [], "clipped_black": [],
           "uniform": [], "mean_range": None, "magenta_suspect": []}
    means = []
    for r in rows:
        p = os.path.join(PROJECT, r["image_path"])
        if not os.path.isfile(p):
            continue
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            g = rgb.convert("L")
            px = list(g.getdata())
            n = len(px)
            mean = sum(px) / n
            means.append(mean)
            if mean < 20:
                res["too_dark"].append([r["filename"], round(mean, 1)])
            if mean > 235:
                res["too_bright"].append([r["filename"], round(mean, 1)])
            white = sum(1 for v in px if v >= 254) / n
            black = sum(1 for v in px if v <= 1) / n
            # metal highlights are legitimate - only flag gross clipping
            if white > 0.05:
                res["clipped_white"].append([r["filename"], round(white, 4)])
            if black > 0.05:
                res["clipped_black"].append([r["filename"], round(black, 4)])
            if max(px) - min(px) < 10:
                res["uniform"].append(r["filename"])
            # a missing texture renders as flat magenta
            small = rgb.resize((64, 64))
            mag = sum(1 for (rr, gg, bb) in small.getdata()
                      if rr > 180 and bb > 180 and gg < 80) / 4096.0
            if mag > 0.10:
                res["magenta_suspect"].append([r["filename"], round(mag, 3)])
    res["mean_range"] = [round(min(means), 1), round(max(means), 1)] if means else None
    return res


def check_variation(rows):
    """The five renders of a material must not share identical randomization."""
    res = {"no_variation": [], "duplicate_states": [], "per_material": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[r["material_id"]].append(r)

    fields = {
        "camera": ("camera_location_x", "camera_location_y", "camera_location_z"),
        "light_pos": ("light_location_x", "light_location_y", "light_location_z"),
        "light_power": ("light_power",),
        "obj_rot": ("object_rotation_x_deg", "object_rotation_y_deg", "object_rotation_z_deg"),
    }
    for mid, grp in groups.items():
        summary = {}
        states = set()
        for r in grp:
            states.add(tuple(r[f] for fs in fields.values() for f in fs))
        if len(states) != len(grp):
            res["duplicate_states"].append(mid)
        for label, fs in fields.items():
            spreads = []
            for f in fs:
                vals = [float(r[f]) for r in grp]
                spreads.append(round(max(vals) - min(vals), 4))
            summary[label] = spreads
            if all(s == 0 for s in spreads):
                res["no_variation"].append([mid, label])
        res["per_material"][mid] = summary
    return res


def contact_sheet(rows, thumb=200, pad=22):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    for c in by_class:
        by_class[c].sort(key=lambda r: int(r["render_index"]))

    cols = EXPECTED_PER_CLASS
    w = thumb * cols
    h = (thumb + pad) * len(CLASS_ORDER)
    sheet = Image.new("RGB", (w, h), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for row, cls in enumerate(CLASS_ORDER):
        y = row * (thumb + pad)
        items = by_class.get(cls, [])
        label = "%s   (%s)" % (cls.upper(), items[0]["material_id"] if items else "-")
        d.text((5, y + 6), label, fill=(235, 235, 235))
        for col, r in enumerate(items[:cols]):
            p = os.path.join(PROJECT, r["image_path"])
            if os.path.isfile(p):
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((thumb, thumb)),
                                (col * thumb, y + pad))
    sheet.save(CONTACT_SHEET)
    return CONTACT_SHEET


def main():
    rows = _rows()
    split_map = _split_lookup()
    report = {
        "n_manifest_rows": len(rows),
        "files": check_files(rows),
        "materials": check_materials(rows, split_map),
        "framing": check_framing(rows),
        "exposure": check_exposure(rows),
        "variation": check_variation(rows),
    }
    report["contact_sheet"] = contact_sheet(rows)
    return report


if __name__ == "__main__":
    import json
    rep = main()

    f, m, fr, ex, va = (rep["files"], rep["materials"], rep["framing"],
                        rep["exposure"], rep["variation"])
    checks = [
        ("exactly 35 PNG files", f["png_count"] == EXPECTED_TOTAL, f["png_count"]),
        ("manifest has 35 rows", rep["n_manifest_rows"] == EXPECTED_TOTAL, rep["n_manifest_rows"]),
        ("5 per class, all 7 classes",
         sorted(f["per_class"]) == sorted(CLASS_ORDER)
         and all(v == EXPECTED_PER_CLASS for v in f["per_class"].values()), f["per_class"]),
        ("no filename collisions", not f["collisions"], f["collisions"] or "none"),
        ("no missing files", not f["missing"], f["missing"] or "none"),
        ("all 512x512", not f["wrong_size"], f["wrong_size"] or "ok"),
        ("no zero-byte files", not f["zero_byte"], f["zero_byte"] or "none"),
        ("no corrupt files", not f["corrupt"], f["corrupt"] or "none"),
        ("material on object matches intended", not m["material_mismatch"],
         m["material_mismatch"] or "all match"),
        ("class matches split manifest", not m["class_mismatch"], m["class_mismatch"] or "ok"),
        ("split matches split manifest", not m["split_mismatch"], m["split_mismatch"] or "ok"),
        ("sphere never cropped", not fr["cropped"], fr["cropped"] or "ok"),
        ("sphere not too small", not fr["too_small"], fr["too_small"] or "ok"),
        ("sphere not too large", not fr["too_large"], fr["too_large"] or "ok"),
        ("ground visible below sphere", not fr["no_ground_below"], fr["no_ground_below"] or "ok"),
        ("contact shadow present", not fr["no_contact_shadow"],
         fr["no_contact_shadow"] or "ok"),
        ("no all-black images", not ex["too_dark"], ex["too_dark"] or "ok"),
        ("no all-white images", not ex["too_bright"], ex["too_bright"] or "ok"),
        ("no gross white clipping", not ex["clipped_white"], ex["clipped_white"] or "ok"),
        ("no gross black clipping", not ex["clipped_black"], ex["clipped_black"] or "ok"),
        ("no uniform / blank images", not ex["uniform"], ex["uniform"] or "ok"),
        ("no missing-texture magenta", not ex["magenta_suspect"], ex["magenta_suspect"] or "ok"),
        ("variation present in every axis", not va["no_variation"], va["no_variation"] or "ok"),
        ("no duplicate randomization states", not va["duplicate_states"],
         va["duplicate_states"] or "ok"),
    ]
    for name, ok, detail in checks:
        print("  [%s] %-38s %s" % ("PASS" if ok else "FAIL", name, detail))
    print()
    print("  sphere frame fraction range: %s (target %s)"
          % (fr["fraction_range"], list(FRAME_FRACTION)))
    print("  image mean luminance range : %s" % ex["mean_range"])
    print("  contact sheet              : %s" % rep["contact_sheet"])
    print()
    print("  ALL CHECKS PASSED" if all(c[1] for c in checks) else "  FAILURES PRESENT")

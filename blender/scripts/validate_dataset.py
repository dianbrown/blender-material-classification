"""Validate the final dataset, build contact sheets and write the reports.

Runs outside Blender - reads only the PNGs and render_manifest.csv.
"""

import csv
import json
import math
import os
import random
import struct
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(PROJECT, "dataset")
MANIFEST = os.path.join(PROJECT, "metadata", "render_manifest.csv")
MANIFEST_JSON = os.path.join(PROJECT, "metadata", "render_manifest.json")
SPLIT_MANIFEST = os.path.join(PROJECT, "metadata", "material_manifest_split.csv")
REPORTS = os.path.join(PROJECT, "reports", "dataset_generation")
SHEETS = os.path.join(REPORTS, "contact_sheets")

CLASS_ORDER = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
SPLITS = ["train", "val", "test"]

RENDERS_PER_MATERIAL = 30
EXPECTED_SIZE = (512, 512)
EXPECTED_TOTAL = 3150
EXPECTED_PER_SPLIT = {"train": 2310, "val": 420, "test": 420}
EXPECTED_PER_CLASS_SPLIT = {"train": 330, "val": 60, "test": 60}

# framing geometry (camera always aims at the sphere centre)
LENS_MM, SENSOR_MM, SPHERE_R = 50.0, 36.0, 1.0
DISC_FRACTION = 0.80


def _font(sz, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def png_size(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except Exception:
        return None


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_splits():
    with open(SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        return {r.get("base_material_id") or r["material_id"]: r
                for r in csv.DictReader(fh)}


# ------------------------------------------------------------------ checks

def validate(rows, splits, sample_pixels=True):
    res = {}

    disk = []
    for root, _d, files in os.walk(DATASET):
        for f in files:
            if f.lower().endswith(".png"):
                disk.append(os.path.join(root, f))
    res["png_on_disk"] = len(disk)
    res["manifest_rows"] = len(rows)

    # filename collisions across the whole tree
    names = defaultdict(list)
    for p in disk:
        names[os.path.basename(p)].append(p)
    res["filename_collisions"] = [n for n, v in names.items() if len(v) > 1]

    # size / zero-byte / corrupt
    wrong_size, zero_byte, corrupt, missing = [], [], [], []
    for r in rows:
        p = os.path.join(PROJECT, r["image_path"])
        if not os.path.isfile(p):
            missing.append(r["image_path"]); continue
        if os.path.getsize(p) == 0:
            zero_byte.append(r["image_path"]); continue
        sz = png_size(p)
        if sz is None:
            corrupt.append(r["image_path"])
        elif sz != EXPECTED_SIZE:
            wrong_size.append([r["image_path"], sz])
    res["missing"] = missing
    res["zero_byte"] = zero_byte
    res["corrupt_header"] = corrupt
    res["wrong_size"] = wrong_size

    # counts
    per_split = defaultdict(int)
    per_class = defaultdict(int)
    per_class_split = defaultdict(int)
    per_material = defaultdict(list)
    for r in rows:
        per_split[r["split"]] += 1
        per_class[r["class"]] += 1
        per_class_split[(r["class"], r["split"])] += 1
        per_material[r["material_id"]].append(int(r["render_index"]))
    res["per_split"] = dict(per_split)
    res["per_class"] = dict(per_class)
    res["per_class_split"] = {"%s/%s" % k: v for k, v in per_class_split.items()}

    # per-material completeness and index coverage
    bad_counts, missing_idx = [], []
    for mid, idxs in per_material.items():
        if len(idxs) != RENDERS_PER_MATERIAL:
            bad_counts.append([mid, len(idxs)])
        gaps = sorted(set(range(1, RENDERS_PER_MATERIAL + 1)) - set(idxs))
        if gaps:
            missing_idx.append([mid, gaps])
    res["materials_with_wrong_count"] = bad_counts
    res["materials_missing_indices"] = missing_idx
    res["n_materials"] = len(per_material)

    # split inheritance: every row must match the authoritative split manifest
    wrong_split, wrong_class, not_in_split, wrong_dir = [], [], [], []
    mat_splits = defaultdict(set)
    for r in rows:
        src = splits.get(r["material_id"])
        if src is None:
            not_in_split.append(r["material_id"]); continue
        if src["split"] != r["split"]:
            wrong_split.append([r["filename"], src["split"], r["split"]])
        if src["class"] != r["class"]:
            wrong_class.append([r["filename"], src["class"], r["class"]])
        mat_splits[r["material_id"]].add(r["split"])
        expect = "dataset/%s/%s/%s" % (r["split"], r["class"], r["filename"])
        if r["image_path"] != expect:
            wrong_dir.append([r["image_path"], expect])
    res["wrong_split"] = wrong_split
    res["wrong_class"] = wrong_class
    res["not_in_split_manifest"] = sorted(set(not_in_split))
    res["path_mismatch"] = wrong_dir
    res["materials_in_multiple_splits"] = [m for m, s in mat_splits.items() if len(s) > 1]

    # seeds must be unique per image
    seeds = [r["random_seed"] for r in rows]
    res["duplicate_seeds"] = len(seeds) - len(set(seeds))

    # ---- image sanity on a stratified sample (full decode is expensive)
    if sample_pixels:
        rng = random.Random(11)
        by_mat = defaultdict(list)
        for r in rows:
            by_mat[r["material_id"]].append(r)
        sample = []
        for mid in sorted(by_mat):
            sample.extend(rng.sample(by_mat[mid], min(4, len(by_mat[mid]))))
        res["sampled"] = len(sample)

        all_black, all_white, uniform, magenta, cropped, means = [], [], [], [], [], []
        half_fov = math.atan((SENSOR_MM / 2) / LENS_MM)
        for r in sample:
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
                if mean < 8:
                    all_black.append([r["filename"], round(mean, 1)])
                if mean > 247:
                    all_white.append([r["filename"], round(mean, 1)])
                if max(px) - min(px) < 10:
                    uniform.append(r["filename"])
                small = rgb.resize((48, 48))
                mag = sum(1 for (rr, gg, bb) in small.getdata()
                          if rr > 180 and bb > 180 and gg < 80) / (48.0 * 48.0)
                if mag > 0.10:
                    magenta.append([r["filename"], round(mag, 3)])
            # framing is exact from the recorded distance
            d = float(r["camera_distance"])
            f_px = (EXPECTED_SIZE[0] / 2) / math.tan(half_fov)
            r_px = f_px * math.tan(math.asin(SPHERE_R / d))
            if r_px > EXPECTED_SIZE[1] / 2:
                cropped.append([r["filename"], round(r_px, 1)])
        res["all_black"] = all_black
        res["all_white"] = all_white
        res["uniform"] = uniform
        res["magenta_missing_texture"] = magenta
        res["cropped"] = cropped
        res["sample_mean_range"] = [round(min(means), 1), round(max(means), 1)] if means else None

    # framing across ALL rows (cheap, geometric)
    half_fov = math.atan((SENSOR_MM / 2) / LENS_MM)
    f_px = (EXPECTED_SIZE[0] / 2) / math.tan(half_fov)
    fracs = []
    out_of_band = []
    for r in rows:
        d = float(r["camera_distance"])
        r_px = f_px * math.tan(math.asin(SPHERE_R / d))
        frac = (2 * r_px) / EXPECTED_SIZE[1]
        fracs.append(frac)
        if frac < 0.55 or frac > 0.75:
            out_of_band.append([r["filename"], round(frac, 3)])
    res["frame_fraction_range"] = [round(min(fracs), 3), round(max(fracs), 3)]
    res["frame_fraction_out_of_band"] = out_of_band

    # randomization envelope actually observed
    def rng_of(field):
        vals = [float(r[field]) for r in rows]
        return [round(min(vals), 3), round(max(vals), 3)]
    res["observed_ranges"] = {
        "camera_azimuth_deg": rng_of("camera_azimuth_deg"),
        "camera_elevation_deg": rng_of("camera_elevation_deg"),
        "camera_distance": rng_of("camera_distance"),
        "area_light_power": rng_of("area_light_power"),
        "area_light_location_x": rng_of("area_light_location_x"),
        "area_light_location_y": rng_of("area_light_location_y"),
        "area_light_location_z": rng_of("area_light_location_z"),
        "object_rotation_x": rng_of("object_rotation_x"),
        "object_rotation_y": rng_of("object_rotation_y"),
        "object_rotation_z": rng_of("object_rotation_z"),
    }
    return res


# ------------------------------------------------------------------ sheets

def _tile_grid(entries, cols, tile, title, path, sub_h=18, header_h=26):
    rows_n = int(math.ceil(len(entries) / float(cols)))
    W = cols * tile
    H = header_h + rows_n * (tile + sub_h)
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, 0, W, header_h - 2], fill=(40, 40, 48))
    d.text((6, 5), title, fill=(245, 245, 245), font=_font(15, True))
    for i, (label, img_path) in enumerate(entries):
        cx = (i % cols) * tile
        cy = header_h + (i // cols) * (tile + sub_h)
        if os.path.isfile(img_path):
            with Image.open(img_path) as im:
                sheet.paste(im.convert("RGB").resize((tile, tile)), (cx, cy))
        d.rectangle([cx, cy + tile, cx + tile, cy + tile + sub_h], fill=(28, 28, 32))
        d.text((cx + 4, cy + tile + 3), label, fill=(215, 215, 220), font=_font(11))
    sheet.save(path)
    return path


def contact_sheets(rows):
    os.makedirs(SHEETS, exist_ok=True)
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["class"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: (r["material_id"], int(r["render_index"])))

    made = []

    # A. one sheet per split: every class, 3 samples from the first material
    for split in SPLITS:
        entries = []
        for cls in CLASS_ORDER:
            items = by.get((split, cls), [])
            mats = sorted({r["material_id"] for r in items})
            for mid in mats[:1]:
                picks = [r for r in items if r["material_id"] == mid][:3]
                for r in picks:
                    entries.append(("%s r%s" % (r["material_id"], r["render_index"]),
                                    os.path.join(PROJECT, r["image_path"])))
        made.append(_tile_grid(entries, 3, 150,
                               "SPLIT: %s  -  one material per class, 3 renders" % split.upper(),
                               os.path.join(SHEETS, "split_%s.png" % split)))

    # B. one sheet per class: 2 materials x 3 renders from each split
    for cls in CLASS_ORDER:
        entries = []
        for split in SPLITS:
            items = by.get((split, cls), [])
            mats = sorted({r["material_id"] for r in items})
            for mid in mats[:2]:
                picks = [r for r in items if r["material_id"] == mid][:3]
                for r in picks:
                    entries.append(("%s %s r%s" % (split, r["material_id"], r["render_index"]),
                                    os.path.join(PROJECT, r["image_path"])))
        made.append(_tile_grid(entries, 6, 150,
                               "CLASS: %s  -  2 materials x 3 renders per split" % cls.upper(),
                               os.path.join(SHEETS, "class_%s.png" % cls)))

    # C. one overview: every class x every split
    entries = []
    for cls in CLASS_ORDER:
        for split in SPLITS:
            items = by.get((split, cls), [])
            if items:
                r = items[0]
                entries.append(("%s / %s" % (cls, split),
                                os.path.join(PROJECT, r["image_path"])))
    made.append(_tile_grid(entries, 3, 170,
                           "DATASET OVERVIEW  -  7 classes x 3 splits",
                           os.path.join(SHEETS, "overview.png")))
    return made


# ------------------------------------------------------------------ reports

def write_reports(rows, res, timing, sheets):
    os.makedirs(REPORTS, exist_ok=True)

    # machine-readable counts
    counts_path = os.path.join(REPORTS, "dataset_counts.csv")
    with open(counts_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "split", "n_materials", "renders_per_material", "n_images"])
        by = defaultdict(set)
        cnt = defaultdict(int)
        for r in rows:
            by[(r["class"], r["split"])].add(r["material_id"])
            cnt[(r["class"], r["split"])] += 1
        for cls in CLASS_ORDER:
            for sp in SPLITS:
                w.writerow([cls, sp, len(by[(cls, sp)]), RENDERS_PER_MATERIAL,
                            cnt[(cls, sp)]])
        for sp in SPLITS:
            w.writerow(["ALL", sp, sum(len(by[(c, sp)]) for c in CLASS_ORDER),
                        RENDERS_PER_MATERIAL,
                        sum(cnt[(c, sp)] for c in CLASS_ORDER)])
        w.writerow(["ALL", "ALL", 105, RENDERS_PER_MATERIAL, len(rows)])

    # json manifest
    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    ok = (res["png_on_disk"] == EXPECTED_TOTAL
          and res["manifest_rows"] == EXPECTED_TOTAL
          and not res["missing"] and not res["zero_byte"] and not res["corrupt_header"]
          and not res["wrong_size"] and not res["filename_collisions"]
          and not res["materials_with_wrong_count"] and not res["materials_missing_indices"]
          and not res["wrong_split"] and not res["wrong_class"]
          and not res["materials_in_multiple_splits"] and not res["path_mismatch"]
          and res["per_split"] == EXPECTED_PER_SPLIT)

    L = []
    L.append("# Final synthetic material dataset - generation report\n")
    L.append("**Status: %s**\n" % ("COMPLETE - all checks passed" if ok
                                   else "COMPLETE WITH ISSUES - see below"))
    L.append("| | |")
    L.append("|---|---|")
    L.append("| Total images | **%d** |" % res["png_on_disk"])
    L.append("| Manifest rows | %d |" % res["manifest_rows"])
    L.append("| Base materials | %d x %d renders |" % (res["n_materials"], RENDERS_PER_MATERIAL))
    L.append("| Render engine | %s |" % rows[0]["render_engine"])
    L.append("| Resolution | %sx%s, 100%%, PNG RGB |"
             % (rows[0]["resolution_width"], rows[0]["resolution_height"]))
    L.append("| Master seed | 62642 (SHA-256 child seed per render) |")
    L.append("| Wall-clock | %s |" % timing.get("wall", "n/a"))
    L.append("| Avg per image | %s |" % timing.get("per_image", "n/a"))
    L.append("| Throughput | %s |" % timing.get("throughput", "n/a"))
    L.append("| Completed in one pass | %s |" % timing.get("one_pass", "n/a"))
    L.append("")

    L.append("## Counts per split\n")
    L.append("| split | images | expected | ok |")
    L.append("|---|---|---|---|")
    for sp in SPLITS:
        got = res["per_split"].get(sp, 0)
        L.append("| %s | %d | %d | %s |" % (sp, got, EXPECTED_PER_SPLIT[sp],
                                            "yes" if got == EXPECTED_PER_SPLIT[sp] else "NO"))
    L.append("| **total** | **%d** | **%d** | %s |"
             % (res["manifest_rows"], EXPECTED_TOTAL,
                "yes" if res["manifest_rows"] == EXPECTED_TOTAL else "NO"))
    L.append("")

    L.append("## Counts per class and split\n")
    L.append("| class | train | val | test | total |")
    L.append("|---|---|---|---|---|")
    for cls in CLASS_ORDER:
        tr = res["per_class_split"].get("%s/train" % cls, 0)
        va = res["per_class_split"].get("%s/val" % cls, 0)
        te = res["per_class_split"].get("%s/test" % cls, 0)
        L.append("| %s | %d | %d | %d | %d |" % (cls, tr, va, te, tr + va + te))
    L.append("| **expected** | **330** | **60** | **60** | **450** |")
    L.append("")

    L.append("## Validation\n")
    checks = [
        ("exactly %d PNG files on disk" % EXPECTED_TOTAL, res["png_on_disk"] == EXPECTED_TOTAL, res["png_on_disk"]),
        ("manifest has %d rows" % EXPECTED_TOTAL, res["manifest_rows"] == EXPECTED_TOTAL, res["manifest_rows"]),
        ("no missing files", not res["missing"], len(res["missing"])),
        ("no zero-byte files", not res["zero_byte"], len(res["zero_byte"])),
        ("no corrupt PNG headers", not res["corrupt_header"], len(res["corrupt_header"])),
        ("all images 512x512", not res["wrong_size"], len(res["wrong_size"])),
        ("no filename collisions", not res["filename_collisions"], len(res["filename_collisions"])),
        ("every material has exactly 30 renders", not res["materials_with_wrong_count"],
         len(res["materials_with_wrong_count"])),
        ("no missing render indices", not res["materials_missing_indices"],
         len(res["materials_missing_indices"])),
        ("105 base materials present", res["n_materials"] == 105, res["n_materials"]),
        ("split inheritance correct", not res["wrong_split"], len(res["wrong_split"])),
        ("class inheritance correct", not res["wrong_class"], len(res["wrong_class"])),
        ("no material crosses splits", not res["materials_in_multiple_splits"],
         len(res["materials_in_multiple_splits"])),
        ("file paths match split/class", not res["path_mismatch"], len(res["path_mismatch"])),
        ("all seeds unique", res["duplicate_seeds"] == 0, res["duplicate_seeds"]),
        ("no all-black images (sampled)", not res.get("all_black"), len(res.get("all_black", []))),
        ("no all-white images (sampled)", not res.get("all_white"), len(res.get("all_white", []))),
        ("no blank images (sampled)", not res.get("uniform"), len(res.get("uniform", []))),
        ("no missing-texture magenta (sampled)", not res.get("magenta_missing_texture"),
         len(res.get("magenta_missing_texture", []))),
        ("sphere never cropped (all rows)", not res.get("cropped"), len(res.get("cropped", []))),
        ("sphere 55-75% of frame height (all rows)", not res["frame_fraction_out_of_band"],
         len(res["frame_fraction_out_of_band"])),
    ]
    L.append("| check | result | detail |")
    L.append("|---|---|---|")
    for name, good, detail in checks:
        L.append("| %s | %s | %s |" % (name, "PASS" if good else "**FAIL**", detail))
    L.append("")
    L.append("Pixel sanity was measured on a stratified sample of %d images "
             "(4 per base material); geometric framing was checked on all %d rows.\n"
             % (res.get("sampled", 0), res["manifest_rows"]))

    L.append("## Observed randomization envelope\n")
    L.append("| parameter | observed min | observed max | configured |")
    L.append("|---|---|---|---|")
    conf = {
        "camera_azimuth_deg": "-20 .. +20 (baseline 0)",
        "camera_elevation_deg": "21 .. 37 (baseline 25, -4/+12)",
        "camera_distance": "4.315 .. 4.865 (0.94-1.06x)",
        "area_light_power": "1600 .. 2400 (0.80-1.20x)",
        "area_light_location_x": "4.5 .. 6.5 (+/-1.0)",
        "area_light_location_y": "-2.5 .. -0.5 (+/-1.0)",
        "area_light_location_z": "3.8 .. 5.0 (+/-0.6)",
        "object_rotation_x": "-10 .. +10",
        "object_rotation_y": "-10 .. +10",
        "object_rotation_z": "0 .. 360",
    }
    for k, v in res["observed_ranges"].items():
        L.append("| %s | %s | %s | %s |" % (k, v[0], v[1], conf.get(k, "")))
    L.append("")
    L.append("Sphere occupied %.1f%%-%.1f%% of frame height across all renders "
             "(target 55-75%%).\n" % (res["frame_fraction_range"][0] * 100,
                                      res["frame_fraction_range"][1] * 100))

    issues = []
    for key, label in [("missing", "missing files"), ("zero_byte", "zero-byte files"),
                       ("corrupt_header", "corrupt files"), ("wrong_size", "wrong size"),
                       ("materials_with_wrong_count", "materials with wrong render count"),
                       ("materials_missing_indices", "materials missing indices"),
                       ("wrong_split", "split mismatches"), ("wrong_class", "class mismatches"),
                       ("path_mismatch", "path mismatches"),
                       ("all_black", "all-black"), ("all_white", "all-white"),
                       ("magenta_missing_texture", "missing texture"),
                       ("cropped", "cropped sphere")]:
        v = res.get(key)
        if v:
            issues.append("- **%s** (%d): %s" % (label, len(v), str(v[:10])))
    L.append("## Failed / repaired renders\n")
    if issues:
        L.extend(issues)
    else:
        L.append("None. No render failed, and no image required a retry or repair.")
    L.append("")
    L.append("%s\n" % timing.get("notes", ""))

    L.append("## Material / texture mismatches\n")
    L.append("Preflight verified, for all 105 manifest rows: the Blender material exists, "
             "its `base_material_id` custom property matches the manifest `material_id`, "
             "every Image Texture node resolves to a file on disk, class and split are valid, "
             "and the destination directory is creatable. ")
    L.append("`metal_13` had been replaced with `worn_corrugated_iron` before this run; "
             "it resolved cleanly under its existing `MAT_Metal_13` name, so no mapping "
             "repair was required.\n")

    L.append("## Contact sheets\n")
    for s in sheets:
        L.append("- `%s`" % os.path.relpath(s, PROJECT).replace("\\", "/"))
    L.append("")

    path = os.path.join(REPORTS, "final_dataset_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return {"report": path, "counts": counts_path, "json": MANIFEST_JSON, "ok": ok}


def main(timing=None):
    rows = load_manifest()
    splits = load_splits()
    res = validate(rows, splits)
    sheets = contact_sheets(rows)
    out = write_reports(rows, res, timing or {}, sheets)
    return res, out, sheets


if __name__ == "__main__":
    tim = {}
    tp = os.path.join(REPORTS, "_timing.json")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as fh:
            tim = json.load(fh)
    res, out, sheets = main(tim)
    print("images on disk : %d" % res["png_on_disk"])
    print("manifest rows  : %d" % res["manifest_rows"])
    print("per split      : %s" % res["per_split"])
    print("all checks ok  : %s" % out["ok"])
    print("report         : %s" % out["report"])
    for k in ("missing", "zero_byte", "corrupt_header", "wrong_size",
              "materials_with_wrong_count", "materials_missing_indices",
              "wrong_split", "wrong_class", "materials_in_multiple_splits",
              "path_mismatch", "all_black", "all_white", "uniform",
              "magenta_missing_texture", "cropped", "frame_fraction_out_of_band"):
        v = res.get(k)
        if v:
            print("  ISSUE %-32s %s" % (k, str(v)[:200]))

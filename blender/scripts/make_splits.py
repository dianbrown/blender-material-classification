"""Leakage-safe train/val/test assignment at the base-material level.

Reads material_manifest.csv, assigns each base material (never an individual
render) to exactly one split, and writes the split manifests plus a summary.

Reproducibility: a single random.Random(SEED) is consumed in a fixed order --
classes in CLASS_ORDER, base materials sorted by base_material_id within each
class -- so repeated runs produce byte-identical output.
"""

import csv
import json
import os
import random

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_IN = os.path.join(PROJECT, "metadata", "material_manifest.csv")
CSV_OUT = os.path.join(PROJECT, "metadata", "material_manifest_split.csv")
JSON_OUT = os.path.join(PROJECT, "metadata", "material_manifest_split.json")
SUMMARY_OUT = os.path.join(PROJECT, "metadata", "dataset_split_summary.csv")

SEED = 42
CLASS_ORDER = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]

# Materials per class, per split. Order matters: it is the draw order.
QUOTA = [("train", 11), ("val", 2), ("test", 2)]

RENDERS_PER_MATERIAL = 30

BOOL_FIELDS = {"has_displacement", "has_ao", "used_default_roughness", "needs_review"}


def load_manifest():
    with open(MANIFEST_IN, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames)


def id_column(fieldnames):
    """The base-material identity column.

    material_manifest.csv stores it as 'material_id'; it carries exactly the
    value of the Blender material's 'base_material_id' custom property.
    """
    for candidate in ("base_material_id", "material_id"):
        if candidate in fieldnames:
            return candidate
    raise KeyError("no base-material identity column in %s" % MANIFEST_IN)


def assign_splits(rows, idcol):
    """Return {base_material_id: split}. One draw per class, fixed order."""
    rng = random.Random(SEED)

    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r[idcol])

    assignment = {}
    for cls in CLASS_ORDER:
        ids = sorted(by_class.get(cls, []))
        shuffled = list(ids)
        rng.shuffle(shuffled)

        cursor = 0
        for split, n in QUOTA:
            for mid in shuffled[cursor:cursor + n]:
                assignment[mid] = split
            cursor += n
        if cursor != len(shuffled):
            raise ValueError(
                "class %s has %d base materials; quota covers %d"
                % (cls, len(shuffled), cursor))
    return assignment


def write_manifests(rows, fieldnames):
    out_fields = list(fieldnames) + ["split"]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    # JSON keeps real booleans rather than the CSV's "True"/"False" strings.
    json_rows = []
    for r in rows:
        obj = dict(r)
        for f in BOOL_FIELDS:
            if f in obj:
                obj[f] = str(obj[f]).strip().lower() == "true"
        json_rows.append(obj)
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(json_rows, fh, indent=1)


def write_summary(rows):
    counts = {}
    for r in rows:
        counts[(r["class"], r["split"])] = counts.get((r["class"], r["split"]), 0) + 1

    out = []
    for cls in CLASS_ORDER:
        for split, _ in QUOTA:
            n = counts.get((cls, split), 0)
            out.append({
                "class": cls,
                "split": split,
                "number_of_base_materials": n,
                "expected_renders_per_material": RENDERS_PER_MATERIAL,
                "expected_images": n * RENDERS_PER_MATERIAL,
            })
    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "class", "split", "number_of_base_materials",
            "expected_renders_per_material", "expected_images"])
        w.writeheader()
        w.writerows(out)
    return out


def validate(rows, summary, idcol):
    checks = []

    def check(num, desc, ok, detail=""):
        checks.append({"n": num, "check": desc, "pass": bool(ok), "detail": detail})

    # 1. exactly 105 materials assigned
    assigned = [r for r in rows if r.get("split")]
    check(1, "exactly 105 materials assigned", len(assigned) == 105,
          "assigned=%d of %d rows" % (len(assigned), len(rows)))

    # 2. every material has exactly one split, from the allowed vocabulary
    allowed = {"train", "val", "test"}
    bad = [r["blender_material_name"] for r in rows if r.get("split") not in allowed]
    check(2, "every material has exactly one valid split value", not bad,
          "invalid=%s" % (bad or "none"))

    # 3. no base_material_id in more than one split
    seen = {}
    dupes = []
    for r in rows:
        mid = r[idcol]
        if mid in seen and seen[mid] != r["split"]:
            dupes.append(mid)
        elif mid in seen:
            dupes.append(mid + " (duplicate row)")
        seen[mid] = r["split"]
    check(3, "no base_material_id appears in more than one split", not dupes,
          "unique ids=%d, conflicts=%s" % (len(seen), dupes or "none"))

    # 4. per-class quota
    per = {}
    for r in rows:
        per.setdefault(r["class"], {}).setdefault(r["split"], 0)
        per[r["class"]][r["split"]] += 1
    wrong = {c: v for c, v in per.items()
             if (v.get("train"), v.get("val"), v.get("test")) != (11, 2, 2)}
    check(4, "every class has 11 train / 2 val / 2 test", not wrong,
          "offenders=%s" % (wrong or "none"))

    # 5/6/7. split totals
    tot = {}
    for r in rows:
        tot[r["split"]] = tot.get(r["split"], 0) + 1
    check(5, "train contains 77 materials", tot.get("train") == 77,
          "train=%s" % tot.get("train"))
    check(6, "validation contains 14 materials", tot.get("val") == 14,
          "val=%s" % tot.get("val"))
    check(7, "test contains 14 materials", tot.get("test") == 14,
          "test=%s" % tot.get("test"))

    # 8. expected image count
    total_images = sum(s["expected_images"] for s in summary)
    check(8, "expected final image count equals 3150", total_images == 3150,
          "total=%d (train=%d val=%d test=%d)" % (
              total_images,
              tot.get("train", 0) * RENDERS_PER_MATERIAL,
              tot.get("val", 0) * RENDERS_PER_MATERIAL,
              tot.get("test", 0) * RENDERS_PER_MATERIAL))
    return checks


def main():
    rows, fieldnames = load_manifest()
    if "split" in fieldnames:
        fieldnames = [f for f in fieldnames if f != "split"]

    idcol = id_column(fieldnames)
    assignment = assign_splits(rows, idcol)
    for r in rows:
        r["split"] = assignment[r[idcol]]

    write_manifests(rows, fieldnames)
    summary = write_summary(rows)
    checks = validate(rows, summary, idcol)
    return rows, summary, checks


if __name__ == "__main__":
    rows, summary, checks = main()
    print("seed=%d  rows=%d" % (SEED, len(rows)))
    print()
    for c in checks:
        print("  [%s] %d. %s -- %s"
              % ("PASS" if c["pass"] else "FAIL", c["n"], c["check"], c["detail"]))
    print()
    print("  all checks passed" if all(c["pass"] for c in checks) else "  FAILURES PRESENT")

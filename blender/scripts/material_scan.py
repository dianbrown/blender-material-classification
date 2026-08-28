"""PBR texture-set scanner for the material-classification dataset.

Pure python (no bpy) so it can be dry-run outside Blender.
Scans <root>/<class>/<material_folder>/ and classifies texture files into
PBR map slots using suffix-position token matching.
"""

import os
import re

CLASSES = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]

CLASS_TITLE = {c: c.capitalize() for c in CLASSES}

# Class defaults used when no metallic map is present.
METALLIC_DEFAULT = {c: (1.0 if c == "metal" else 0.0) for c in CLASSES}

TEXTURE_EXTS = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".webp", ".tga", ".bmp"}

# Tokens that carry no map-type meaning: resolution, format, bit depth.
DROP_TOKENS = {
    "1k", "2k", "4k", "8k", "16k", "512", "1024", "2048", "4096",
    "jpg", "jpeg", "png", "exr", "tif", "tiff", "webp", "tga", "bmp",
    "8", "16", "32", "bit", "srgb",
}

# Single-token keyword -> (map slot, priority). Higher priority wins.
# NOTE: bare "metal" is deliberately absent - it appears in ordinary material
# names (metal_plate_02, rusty_metal_04) and does not denote a metallic map.
SINGLE = {
    "basecolor":   ("base_color", 100),
    "diff":        ("base_color", 90),
    "diffuse":     ("base_color", 90),
    "albedo":      ("base_color", 80),
    "color":       ("base_color", 70),
    "colour":      ("base_color", 70),
    "col":         ("base_color", 70),

    "rough":       ("roughness", 100),
    "roughness":   ("roughness", 100),

    "normalgl":    ("normal", 100),
    "norgl":       ("normal", 100),
    "normaldx":    ("normal", 60),
    "nordx":       ("normal", 60),
    "normal":      ("normal", 80),
    "nor":         ("normal", 80),
    "normalmap":   ("normal", 80),

    "metallic":    ("metallic", 100),
    "metalness":   ("metallic", 100),

    "disp":         ("displacement", 100),
    "displacement": ("displacement", 100),
    "height":       ("displacement", 90),

    "ao":               ("ao", 100),
    "ambientocclusion": ("ao", 100),
    "occlusion":        ("ao", 90),
}

# Two-token combos, checked before single tokens at the same position.
PAIR = {
    ("base", "color"):        ("base_color", 100),
    ("base", "colour"):       ("base_color", 100),
    ("nor", "gl"):            ("normal", 100),
    ("normal", "gl"):         ("normal", 100),
    ("nor", "dx"):            ("normal", 60),
    ("normal", "dx"):         ("normal", 60),
    ("ambient", "occlusion"): ("ao", 100),
}

# Poly Haven colour-variant naming: col, col1, coll1, col_1 ...
COLOR_VARIANT_RE = re.compile(r"^col{1,2}\d*$")

GL_PAIRS = {("nor", "gl"), ("normal", "gl")}
DX_PAIRS = {("nor", "dx"), ("normal", "dx")}


def _tokenize(stem):
    """Split a filename stem into lowercase tokens, honouring camelCase."""
    # "NormalGL" -> "Normal GL", "Leather011" -> "Leather 011"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    parts = re.split(r"[^A-Za-z0-9]+", s)
    return [p.lower() for p in parts if p]


def classify_file(filename):
    """Return (slot, priority, variant) for a texture filename, or None.

    Keywords are searched from the END of the name inwards, so
    'rough_concrete_diff_2k.jpg' resolves to base_color (its 'diff' suffix)
    rather than roughness (its 'rough' prefix).
    """
    stem = os.path.splitext(filename)[0]
    tokens = [t for t in _tokenize(stem) if t not in DROP_TOKENS]
    if not tokens:
        return None

    for i in range(len(tokens) - 1, -1, -1):
        if i >= 1:
            pair = (tokens[i - 1], tokens[i])
            if pair in PAIR:
                slot, prio = PAIR[pair]
                variant = "gl" if pair in GL_PAIRS else ("dx" if pair in DX_PAIRS else None)
                return slot, prio, variant
        tok = tokens[i]
        if tok in SINGLE:
            slot, prio = SINGLE[tok]
            variant = None
            if slot == "normal":
                if tok in ("normalgl", "norgl"):
                    variant = "gl"
                elif tok in ("normaldx", "nordx"):
                    variant = "dx"
            return slot, prio, variant
        if COLOR_VARIANT_RE.match(tok):
            return "base_color", 70, None
    return None


def scan_material_folder(folder):
    """Classify every texture file in one base-material folder."""
    best = {}          # slot -> ((priority, variant_rank), variant, path)
    unrecognised = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in TEXTURE_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            hit = classify_file(fn)
            if hit is None:
                unrecognised.append(path)
                continue
            slot, prio, variant = hit
            # Prefer OpenGL normals over DirectX / unspecified.
            variant_rank = {"gl": 2, None: 1, "dx": 0}[variant]
            key = (prio, variant_rank)
            if slot not in best or key > best[slot][0]:
                best[slot] = (key, variant, path)
    maps = {slot: v[2] for slot, v in best.items()}
    variants = {slot: v[1] for slot, v in best.items()}
    return maps, variants, unrecognised


def descriptive_name(maps, folder):
    """Longest common prefix of the texture basenames - the vendor's own name."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in maps.values()]
    if not stems:
        return os.path.basename(folder.rstrip("/\\"))
    prefix = os.path.commonprefix(stems)
    # Peel off trailing resolution / format tokens: "Leather011_2K-JPG" -> "Leather011"
    while True:
        stripped = re.sub(r"[^A-Za-z0-9]+$", "", prefix)
        stripped = re.sub(r"(?i)[_-]?(\d+k|jpg|jpeg|png|exr|tiff?)$", "", stripped)
        if stripped == prefix:
            break
        prefix = stripped
    prefix = re.sub(r"[^A-Za-z0-9]+$", "", prefix)
    return prefix or os.path.basename(folder.rstrip("/\\"))


def _folder_number(name, fallback):
    m = re.search(r"(\d+)\s*$", name)
    return "%02d" % (int(m.group(1)) if m else fallback)


def scan_root(root):
    """Scan the whole material library. Returns (records, ignored_dirs)."""
    records = []
    ignored = []
    for entry in sorted(os.listdir(root)):
        class_dir = os.path.join(root, entry)
        if not os.path.isdir(class_dir):
            continue
        cls = entry.lower()
        if cls not in CLASSES:
            ignored.append(class_dir + "  (not a dataset class)")
            continue
        subs = sorted(d for d in os.listdir(class_dir)
                      if os.path.isdir(os.path.join(class_dir, d)))
        if not subs:
            ignored.append(class_dir + "  (class folder has no material subfolders)")
        for idx, sub in enumerate(subs, start=1):
            folder = os.path.join(class_dir, sub)
            maps, variants, unrecognised = scan_material_folder(folder)
            missing = [s for s in ("base_color", "roughness", "normal") if s not in maps]
            records.append({
                "dataset_class": cls,
                "base_material_id": sub,
                "material_name": "MAT_%s_%s" % (CLASS_TITLE[cls], _folder_number(sub, idx)),
                "source_folder": folder.replace("\\", "/") + "/",
                "source_material_name": descriptive_name(maps, folder),
                "maps": maps,
                "normal_variant": variants.get("normal"),
                "missing_maps": missing,
                "unrecognised_files": unrecognised,
                "needs_review": "base_color" in missing,
                "used_default_roughness": "roughness" in missing,
            })
    return records, ignored

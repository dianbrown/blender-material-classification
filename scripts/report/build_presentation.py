"""Build the classroom deck from saved evidence only. No fitting or inference.

Requires Python 3, lxml and Microsoft PowerPoint on Windows. Verification also
requires PyMuPDF. PowerPoint creates native text and tables, and renders the
native Office charts added with their embedded workbooks by this builder.
Run from any directory. Optional --data-root locates the canonical render corpus.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/presentation"
RAW = ROOT / "reports/results/v3"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(text.encode("utf-8"))


def add_native_charts(slides):
    """Add editable DrawingML charts with embedded source workbooks.

    PowerPoint COM chart creation is unavailable in the hidden desktop session.
    The package uses standard OOXML chart parts and literal workbook cells, then
    is opened and rendered by PowerPoint itself to validate interoperability.
    """
    ns = {
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
        "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    }
    for prefix in ["c", "a", "p", "r"]:
        ET.register_namespace(prefix, ns[prefix])

    def el(parent, tag, attrs=None, text=None):
        prefix, local = tag.split(":")
        e = ET.SubElement(parent, "{" + ns[prefix] + "}" + local, attrs or {})
        if text is not None:
            e.text = str(text)
        return e

    def root(tag):
        prefix, local = tag.split(":")
        return ET.Element(
            "{" + ns[prefix] + "}" + local,
            nsmap={None if prefix in ["ct", "rel", "s"] else prefix: ns[prefix]},
        )

    def xml(e):
        return ET.tostring(e, encoding="utf-8", xml_declaration=True)

    def val(parent, tag, value):
        return el(parent, "c:" + tag, {"val": str(value)})

    def workbook(item):
        wb = root("s:workbook")
        ss = el(wb, "s:sheets")
        el(ss, "s:sheet", {"name": "Data", "sheetId": "1", "{" + ns["r"] + "}id": "rId1"})
        ws = root("s:worksheet")
        sd = el(ws, "s:sheetData")
        cells = [["Epoch / condition"] + [x["name"] for x in item["series"]]]
        cells += [
            [str(cat)] + [s["values"][i] if i < len(s["values"]) else None for s in item["series"]]
            for i, cat in enumerate(item["categories"])
        ]
        for i, rr in enumerate(cells, 1):
            row = el(sd, "s:row", {"r": str(i)})
            for j, v in enumerate(rr):
                if v is None:
                    continue
                attrs = {"r": chr(65 + j) + str(i)}
                if isinstance(v, str):
                    attrs["t"] = "inlineStr"
                    cell = el(row, "s:c", attrs)
                    el(el(cell, "s:is"), "s:t", text=v)
                else:
                    el(el(row, "s:c", attrs), "s:v", text=v)
        rel = root("rel:Relationships")
        el(
            rel,
            "rel:Relationship",
            {"Id": "rId1", "Type": ns["r"] + "/worksheet", "Target": "worksheets/sheet1.xml"},
        )
        top = root("rel:Relationships")
        el(
            top,
            "rel:Relationship",
            {"Id": "rId1", "Type": ns["r"] + "/officeDocument", "Target": "xl/workbook.xml"},
        )
        ct = root("ct:Types")
        el(
            ct,
            "ct:Default",
            {
                "Extension": "rels",
                "ContentType": "application/vnd.openxmlformats-package.relationships+xml",
            },
        )
        el(ct, "ct:Default", {"Extension": "xml", "ContentType": "application/xml"})
        el(
            ct,
            "ct:Override",
            {
                "PartName": "/xl/workbook.xml",
                "ContentType": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
                ),
            },
        )
        el(
            ct,
            "ct:Override",
            {
                "PartName": "/xl/worksheets/sheet1.xml",
                "ContentType": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
                ),
            },
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for path, data in [
                ("xl/workbook.xml", wb),
                ("xl/worksheets/sheet1.xml", ws),
                ("xl/_rels/workbook.xml.rels", rel),
                ("_rels/.rels", top),
                ("[Content_Types].xml", ct),
            ]:
                z.writestr(path, xml(data))
        return buf.getvalue()

    def rich(parent, text, size=1400):
        rich = el(parent, "c:rich")
        el(rich, "a:bodyPr")
        el(rich, "a:lstStyle")
        p = el(rich, "a:p")
        run = el(p, "a:r")
        props = el(run, "a:rPr", {"lang": "en-US", "sz": str(size)})
        el(props, "a:latin", {"typeface": "Aptos"})
        el(run, "a:t", text=text)

    def chart_xml(item):
        space = root("c:chartSpace")
        val(space, "date1904", 0)
        val(space, "lang", "en-US")
        ch = el(space, "c:chart")
        title = el(ch, "c:title")
        rich(el(title, "c:tx"), item["title"], 1700)
        val(title, "overlay", 0)
        plot = el(ch, "c:plotArea")
        el(plot, "c:layout")
        typ = el(plot, "c:barChart" if item["chart_type"] == "bar" else "c:lineChart")
        if item["chart_type"] == "bar":
            val(typ, "barDir", "col")
            val(typ, "grouping", "clustered")
        else:
            val(typ, "grouping", "standard")
        val(typ, "varyColors", 0)
        n = len(item["categories"])
        for i, s in enumerate(item["series"]):
            n = len(s["values"])
            ser = el(typ, "c:ser")
            val(ser, "idx", i)
            val(ser, "order", i)
            tx = el(ser, "c:tx")
            ref = el(tx, "c:strRef")
            el(ref, "c:f", text=f"Data!${chr(66 + i)}$1")
            cache = el(ref, "c:strCache")
            val(cache, "ptCount", 1)
            el(el(cache, "c:pt", {"idx": "0"}), "c:v", text=s["name"])
            sp = el(ser, "c:spPr")
            if item["chart_type"] == "bar":
                el(el(sp, "a:solidFill"), "a:srgbClr", {"val": s["color"]})
            line = el(sp, "a:ln", {"w": "25400"})
            el(el(line, "a:solidFill"), "a:srgbClr", {"val": s["color"]})
            if s.get("dash"):
                el(line, "a:prstDash", {"val": "dash"})
            if item["chart_type"] == "line":
                marker = el(ser, "c:marker")
                val(marker, "symbol", "none")
            cat = el(el(ser, "c:cat"), "c:strRef")
            el(cat, "c:f", text=f"Data!$A$2:$A${n + 1}")
            cache = el(cat, "c:strCache")
            val(cache, "ptCount", n)
            for j, c in enumerate(item["categories"][:n]):
                el(el(cache, "c:pt", {"idx": str(j)}), "c:v", text=c)
            ref = el(el(ser, "c:val"), "c:numRef")
            el(ref, "c:f", text=f"Data!${chr(66 + i)}$2:${chr(66 + i)}${n + 1}")
            cache = el(ref, "c:numCache")
            el(cache, "c:formatCode", text="General")
            val(cache, "ptCount", n)
            for j, v in enumerate(s["values"]):
                el(el(cache, "c:pt", {"idx": str(j)}), "c:v", text=v)
            if item["chart_type"] == "line":
                val(ser, "smooth", 0)
        if item["chart_type"] == "bar":
            val(typ, "gapWidth", 90)
            val(typ, "overlap", 0)
        val(typ, "axId", 100)
        val(typ, "axId", 200)
        for axis, id_, other, pos in [("catAx", 100, 200, "b"), ("valAx", 200, 100, "l")]:
            ax = el(plot, "c:" + axis)
            val(ax, "axId", id_)
            scale = el(ax, "c:scaling")
            val(scale, "orientation", "minMax")
            if axis == "valAx":
                if item["ymax"] is not None:
                    val(scale, "max", item["ymax"])
                val(scale, "min", item["ymin"])
            val(ax, "delete", 0)
            val(ax, "axPos", pos)
            if axis == "valAx":
                el(ax, "c:majorGridlines")
                title = el(ax, "c:title")
                rich(el(title, "c:tx"), item["ylabel"], 1100)
                val(title, "overlay", 0)
                el(
                    ax,
                    "c:numFmt",
                    {
                        "formatCode": "0.0" if item["chart_type"] == "line" else "0",
                        "sourceLinked": "0",
                    },
                )
            elif item["chart_type"] == "line":
                title = el(ax, "c:title")
                rich(el(title, "c:tx"), "Epoch", 1100)
                val(title, "overlay", 0)
            val(ax, "majorTickMark", "none")
            val(ax, "minorTickMark", "none")
            val(ax, "tickLblPos", "nextTo")
            val(ax, "crossAx", other)
            val(ax, "crosses", "autoZero")
            if axis == "catAx":
                val(ax, "auto", 1)
                val(ax, "lblAlgn", "ctr")
                val(ax, "lblOffset", 100)
            else:
                val(ax, "crossBetween", "between")
        legend = el(ch, "c:legend")
        val(legend, "legendPos", "b")
        el(legend, "c:layout")
        val(legend, "overlay", 0)
        val(ch, "plotVisOnly", 1)
        val(ch, "dispBlanksAs", "gap")
        tx = el(space, "c:txPr")
        el(tx, "a:bodyPr")
        el(tx, "a:lstStyle")
        p = el(tx, "a:p")
        pp = el(p, "a:pPr")
        d = el(pp, "a:defRPr", {"sz": "1100"})
        el(d, "a:latin", {"typeface": "Aptos"})
        external = el(space, "c:externalData", {"{" + ns["r"] + "}id": "rId1"})
        val(external, "autoUpdate", 0)
        return xml(space)

    with zipfile.ZipFile(OUT / "qa/base.pptx") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    content = ET.fromstring(parts["[Content_Types].xml"])
    if not any(e.get("Extension") == "xlsx" for e in content):
        el(
            content,
            "ct:Default",
            {
                "Extension": "xlsx",
                "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            },
        )
    chart_index = 0
    for i, slide in enumerate(slides, 1):
        items = [it for it in slide["items"] if it["kind"] == "chart"]
        if not items:
            continue
        spath = f"ppt/slides/slide{i}.xml"
        rpath = f"ppt/slides/_rels/slide{i}.xml.rels"
        st = ET.fromstring(parts[spath])
        rels = ET.fromstring(parts[rpath])
        tree = st.find(".//p:spTree", ns)
        for item in items:
            chart_index += 1
            rid = "rIdChart" + str(chart_index)
            frame = el(tree, "p:graphicFrame")
            nv = el(frame, "p:nvGraphicFramePr")
            el(nv, "p:cNvPr", {"id": str(1000 + chart_index), "name": item["title"]})
            el(nv, "p:cNvGraphicFramePr")
            el(nv, "p:nvPr")
            x = el(frame, "p:xfrm")
            el(x, "a:off", {"x": str(item["x"] * 12700), "y": str(item["y"] * 12700)})
            el(x, "a:ext", {"cx": str(item["w"] * 12700), "cy": str(item["h"] * 12700)})
            gd = el(el(frame, "a:graphic"), "a:graphicData", {"uri": ns["c"]})
            el(gd, "c:chart", {"{" + ns["r"] + "}id": rid})
            el(
                rels,
                "rel:Relationship",
                {
                    "Id": rid,
                    "Type": ns["r"] + "/chart",
                    "Target": f"../charts/chart{chart_index}.xml",
                },
            )
            parts[f"ppt/charts/chart{chart_index}.xml"] = chart_xml(item)
            cr = root("rel:Relationships")
            el(
                cr,
                "rel:Relationship",
                {
                    "Id": "rId1",
                    "Type": ns["r"] + "/package",
                    "Target": f"../embeddings/chart{chart_index}.xlsx",
                },
            )
            parts[f"ppt/charts/_rels/chart{chart_index}.xml.rels"] = xml(cr)
            parts[f"ppt/embeddings/chart{chart_index}.xlsx"] = workbook(item)
            el(
                content,
                "ct:Override",
                {
                    "PartName": f"/ppt/charts/chart{chart_index}.xml",
                    "ContentType": (
                        "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"
                    ),
                },
            )
        parts[spath] = xml(st)
        parts[rpath] = xml(rels)
    parts["[Content_Types].xml"] = xml(content)
    core = ET.fromstring(parts["docProps/core.xml"])
    for element in core:
        if element.tag.endswith(("}creator", "}lastModifiedBy")):
            element.text = ""
    parts["docProps/core.xml"] = xml(core)
    with zipfile.ZipFile(OUT / "material_classification.pptx", "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)


def verify(skill_dir):
    import pymupdf

    deck = read(OUT / "deck.json")
    assert len(deck["slides"]) == 14
    assert sum(s["seconds"] for s in deck["slides"]) == 600
    for s in deck["slides"]:
        for src in s["sources"]:
            if src.startswith(("reports/", "metadata/", "configs/", "docs/")):
                assert (ROOT / src.split(",")[0]).is_file(), src
    assert read(OUT / "qa/text_overflow.json") == []
    checks = {}
    commands = {
        "package_integrity": [
            "inspect_presentation_package_integrity.py",
            str(OUT / "material_classification.pptx"),
            "--fail-on-findings",
        ],
        "geometry": [
            "inspect_presentation_layout_geometry.py",
            str(OUT / "material_classification.pptx"),
            "--expected-slide-count",
            "14",
            "--expected-aspect",
            "16:9",
            "--validate-heading-fit",
            "--fail-on-findings",
        ]
        + sum([["--require-native-table-slide", str(i)] for i in [3, 4, 7, 8, 11]], []),
        "native_chart_data": [
            "native_quantitative_chart_gate.py",
            str(OUT / "material_classification.pptx"),
            "--expected-chart-slides",
            "6,10",
            "--require-embedded-workbook-slide",
            "6",
            "--require-embedded-workbook-slide",
            "10",
        ],
        "chart_titles": ["native_chart_title_gate.py", str(OUT / "material_classification.pptx")],
    }
    import sys

    for name, command in commands.items():
        cp = subprocess.run(
            [sys.executable, str(skill_dir / "container_tools" / command[0])] + command[1:],
            capture_output=True,
            text=True,
            check=True,
        )
        checks[name] = json.loads(cp.stdout)
    doc = pymupdf.open(OUT / "material_classification.pdf")
    assert len(doc) == 14 and not doc.metadata.get("author")
    text = "\n".join(p.get_text() for p in doc)
    assert "80.16%" in text and "86.83%" in text and "59.05%" in text
    for i, page in enumerate(doc, 1):
        assert page.rect.width == 960 and page.rect.height == 540
        dest = OUT / "qa/pdf-renders" / f"slide-{i:02}.png"
        dest.parent.mkdir(exist_ok=True)
        page.get_pixmap(matrix=pymupdf.Matrix(1600 / 960, 1600 / 960), alpha=False).save(dest)
        assert (OUT / "renders" / f"slide-{i:02}.png").exists()
    with zipfile.ZipFile(OUT / "material_classification.pptx") as z:
        core = ET.fromstring(z.read("docProps/core.xml"))
        assert all(not e.text for e in core if e.tag.endswith(("}creator", "}lastModifiedBy")))
        assert (
            len(
                [n for n in z.namelist() if n.startswith("ppt/embeddings/") and n.endswith(".xlsx")]
            )
            == 3
        )
    checks["pdf"] = {
        "pages": 14,
        "dimensions_points": [960, 540],
        "author_blank": True,
        "all_pages_rendered": True,
    }
    checks["source_paths"] = {"all_local_note_sources_exist": True}
    checks["artifact_session_import"] = {
        "status": "unavailable",
        "reason": (
            "Workspace dependency loader and artifact-tool package are "
            "unavailable on this Windows host. Native PowerPoint opened and "
            "rendered the final deck instead."
        ),
    }
    write(OUT / "qa/validation.json", json.dumps(checks, indent=2) + "\n")
    print(
        (
            "PASS: evidence paths, 14 slides, native chart workbooks, geometry, "
            "titles, blank author and 14 PDF renders"
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "blender-material-classification/artifacts/v3/holdout_dataset",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=Path.home()
        / (
            ".codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11"
            "618/skills/presentations"
        ),
    )
    args = parser.parse_args()
    if args.verify_only:
        verify(args.skill_dir)
        return
    (OUT / "assets").mkdir(parents=True, exist_ok=True)
    (OUT / "qa").mkdir(exist_ok=True)
    models = ["frozen_selected", "frozen_control", "resnet_fixed", "dinov2_finetuned"]
    names = ["Primary frozen fusion", "ConvNeXt control", "Fixed ResNet-50", "Fine-tuned DINOv2"]
    ev = {m: read(RAW / f"fresh/{m}/evaluation.json") for m in models}
    classes = ev[models[0]]["classes"]
    assert classes == ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
    metrics = {m: ev[m]["image_level"]["metrics"] for m in models}
    expected = [505 / 630, 476 / 630, 489 / 630, 547 / 630]
    for m, accuracy in zip(models, expected):
        assert abs(metrics[m]["accuracy"] - accuracy) < 1e-12
        p = rows(RAW / f"fresh/{m}/predictions.csv")
        assert len(p) == 630 and len({r["source_group"] for r in p}) == 21
    assert read(RAW / "final_model_lock.json")["primary_model"] == "frozen_selected"
    gallery = read(RAW / "report_inputs/gallery_selection.json")
    assets = []

    def copy_image(r, name):
        source = args.data_root / r["image_path"]
        target = OUT / "assets" / f"{name}.png"
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == r["image_sha256"]
            shutil.copyfile(source, target)
        elif not target.exists():
            raise FileNotFoundError(f"Render unavailable: {source}. Supply --data-root.")
        assert hashlib.sha256(target.read_bytes()).hexdigest() == r["image_sha256"]
        assets.append(
            dict(
                path=str(target.relative_to(ROOT)), source=r["image_path"], sha256=r["image_sha256"]
            )
        )
        return str(target.relative_to(OUT)).replace("\\", "/")

    overview = [r for r in gallery if r["gallery"] == "overview"]
    gallery_paths = {r["class"]: copy_image(r, "class_" + r["class"]) for r in overview}
    preds = rows(RAW / "fresh/frozen_selected/predictions.csv")
    examples = [r for r in overview if r["class"] in ["wood", "metal"]]
    examples += [
        next(r for r in gallery if r["gallery"] == "failure" and r["class"] == c)
        for c in ["ceramic", "leather"]
    ]
    ex_paths = [copy_image(r, "example_" + str(i + 1)) for i, r in enumerate(examples)]
    for r in examples:
        matched = next(p for p in preds if p["image_path"] == r["image_path"])
        assert matched["y_pred"] == r["y_pred"] and matched["y_true"] == r["y_true"]
    slides = []

    def slide(title, seconds, notes, sources):
        s = dict(title=title, seconds=seconds, notes=notes, sources=sources, items=[])
        slides.append(s)
        return s

    def text(s, value, x, y, w, h, size=22, color="20364A", bold=False):
        s["items"].append(
            dict(kind="text", text=value, x=x, y=y, w=w, h=h, size=size, color=color, bold=bold)
        )

    def image(s, path, x, y, w, h):
        s["items"].append(dict(kind="image", path=path, x=x, y=y, w=w, h=h))

    def table(s, data, x, y, w, h, widths=None, size=17, heat=False):
        s["items"].append(
            dict(kind="table", data=data, x=x, y=y, w=w, h=h, widths=widths, size=size, heat=heat)
        )

    def chart(s, title, categories, series, x, y, w, h, kind="line", ymin=0, ymax=None, ylabel=""):
        s["items"].append(
            dict(
                kind="chart",
                title=title,
                categories=categories,
                series=series,
                x=x,
                y=y,
                w=w,
                h=h,
                chart_type=kind,
                ymin=ymin,
                ymax=ymax,
                ylabel=ylabel,
            )
        )

    def source(*paths):
        return [str(p).replace("\\", "/") for p in paths]

    s = slide(
        "Recognising materials\nin Blender renders",
        20,
        (
            "Introduce the problem as visual appearance classification in a "
            "controlled rendering setting. The practical aim is to assist "
            "material browsing or organisation. Leave the group-member fields "
            "blank until the presenters supply their names. This rehearsal plan "
            "is about ten minutes, not a lecturer-assigned duration."
        ),
        [
            "ITRI626 practical project brief, sections 2, 10 and 12.2",
            "reports/v3/report.md, Abstract and section 1",
        ],
    )
    text(s, "ITRI626  /  Deep learning for image classification", 48, 62, 800, 32, 19, "137F82")
    text(
        s,
        "Seven classes. Three model families.\nAssessment on unseen source groups.",
        48,
        285,
        670,
        78,
        25,
    )
    text(s, "Group members:", 48, 419, 170, 28, 18)
    # Intentionally empty editable author field. No identity is inferred.
    s["items"].append(
        dict(
            kind="text",
            text="",
            x=220,
            y=416,
            w=680,
            h=60,
            size=20,
            color="20364A",
            bold=False,
            name="Group members - intentionally blank",
        )
    )

    s = slide(
        "The problem and seven classes",
        40,
        (
            "An image receives one of seven appearance labels. This could help "
            "search a rendered asset library. The categories are wood, metal, "
            "stone, concrete, ceramic, fabric and leather. Each picture is an "
            "actual held-out render selected by path, without filtering by "
            "correctness. The same sphere makes texture and reflectance "
            "informative, but this does not identify physical composition. In the"
            " development collection, ceramic includes mosaic or terrazzo-like "
            "appearances and concrete includes plaster or painted walls."
        ),
        [
            "reports/results/v3/report_inputs/gallery_selection.json",
            "reports/v3/report.md, sections 1-2",
        ],
    )
    text(
        s,
        "Classify material appearance to support browsing of rendered assets",
        48,
        105,
        860,
        42,
        23,
    )
    for i, c in enumerate(classes):
        x = 48 + i * 126
        image(s, gallery_paths[c], x, 183, 116, 116)
        text(s, c.title(), x, 313, 120, 27, 18, bold=True)
    text(s, "One RGB image, one label", 48, 385, 800, 34, 26, bold=True)
    text(s, "Appearance categories do not establish physical composition.", 48, 432, 860, 40, 20)

    s = slide(
        "Dataset provenance and group separation",
        50,
        (
            "The study has 3,780 balanced RGB images at 512 by 512 pixels, with "
            "30 rendered views per material. Development contains 450 images per "
            "class and the fresh holdout 90 per class. Source textures come from "
            "Poly Haven and ambientCG under CC0. They include scanned and "
            "procedural assets, so we do not describe all sources as independent "
            "physical scans. Labels follow audited asset identity and appearance."
            " Known generator relatives are merged into groups. Thirty views of a"
            " texture remain correlated. No known development-to-holdout family "
            "or exact-map reuse remains, but unknown physical ancestry cannot be "
            "ruled out. The rendered-dataset release licence is still an owner "
            "decision."
        ),
        [
            "reports/results/v3/data_audit.json",
            "metadata/v3_holdout/source_identity_audit.json",
            "reports/results/v3/report_inputs/study_counts.csv",
            "https://polyhaven.com/license",
            "https://docs.ambientcg.com/license/",
        ],
    )
    table(
        s,
        [
            ["Population", "Images", "Material IDs", "Source groups"],
            ["Development", "3,150", "105", "101"],
            ["Fresh test", "630", "21", "21"],
        ],
        48,
        115,
        864,
        160,
        [0.33, 0.20, 0.23, 0.24],
        21,
    )
    text(
        s,
        "30 views per material. Each class has 450 development + 90 test images.",
        48,
        296,
        864,
        45,
        20,
    )
    text(
        s,
        (
            "Poly Haven + ambientCG CC0 texture assets\nScanned and procedural "
            "sources rendered in Blender"
        ),
        48,
        358,
        864,
        73,
        23,
    )
    text(
        s,
        "Known relatives stay together. Unknown physical ancestry remains a caveat.",
        48,
        456,
        864,
        40,
        19,
        "A25524",
    )

    s = slide(
        "Three model families, four pipelines",
        50,
        (
            "ConvNeXt is a modern convolutional model, ResNet is a residual "
            "convolutional network, and DINOv2 uses a vision transformer. All use"
            " pretrained weights. A frozen ConvNeXt control and fixed ResNet "
            "baseline provide family comparisons. The primary fuses DINOv2 and "
            "ConvNeXt features and fits a scaled, class-balanced logistic "
            "classifier. DINO pools CLS, patch mean and patch standard deviation "
            "at two resolutions, excluding register tokens from patch pooling. "
            "The partially fine-tuned DINO model is exploratory. These are "
            "complete pipeline comparisons because pretraining data, resolution "
            "and compute budgets differ."
        ),
        [
            "reports/results/v3/frozen_initial/protocol.json",
            "configs/v3/finetune.json",
            "reports/results/v3/resnet_baseline/config.json",
            "reports/v3/report.md, section 5",
            "https://arxiv.org/abs/2304.07193",
            "https://arxiv.org/abs/2201.03545",
            "https://arxiv.org/abs/1512.03385",
        ],
    )
    table(
        s,
        [
            ["Pipeline", "Transfer learning", "Input"],
            [
                "Primary fusion",
                "Frozen DINOv2 + ConvNeXt\n3,072 features, logistic head",
                "224 + 336",
            ],
            ["ConvNeXt control", "Frozen ConvNeXt-Tiny\nLogistic head", "224"],
            ["Fixed ResNet-50", "ImageNet weights\nAdapt layer4 and classifier", "224"],
            [
                "Fine-tuned DINOv2",
                "Pretrained ViT-S/14 + registers\nAdapt last two blocks and head",
                "336",
            ],
        ],
        48,
        112,
        864,
        322,
        [0.28, 0.54, 0.18],
        19,
    )
    text(
        s,
        "Architecture, pretraining, resolution and adaptation budgets differ.",
        48,
        459,
        860,
        42,
        20,
        "A25524",
    )

    s = slide(
        "Experimental setup",
        55,
        (
            "The frozen selection uses five outer source-group folds and three "
            "inner grouped folds, seed 20261027. Inner folds select five "
            "representations and C in 0.01, 0.1, 1 and 10 by Macro F1. The scaler"
            " and classifier are fitted inside each training partition. All "
            "original images are development data because the historical test had"
            " already been consulted. The neural comparison uses the same "
            "development pool and the same final holdout. DINO searches two "
            "learning rates in a bounded three-fold exploratory follow-up. "
            "Preprocessing converts to RGB, uses model-specific resize and centre"
            " crop and pretrained normalisation. Neural training alone adds small"
            " random crops, horizontal flips and mild colour jitter. ResNet uses "
            "AdamW with a cosine schedule for ten fixed epochs. DINO uses AdamW "
            "and development early stopping, then a fixed five-epoch final fit. "
            "Both have effective batch size 32, label-smoothed cross entropy and "
            "weight decay 0.01. Development Macro F1 selected the frozen primary "
            "before any fresh scores. Four fixed models then received one clean "
            "evaluation each."
        ),
        [
            "reports/results/v3/frozen_initial/protocol.json",
            "configs/v3/finetune.json",
            "reports/results/v3/resnet_baseline/config.json",
            "reports/results/v3/model_choice.json",
            "reports/results/v3/final_model_lock.json",
        ],
    )
    text(s, "Development and selection", 48, 112, 420, 32, 23, bold=True)
    text(
        s,
        (
            "5 outer / 3 inner grouped folds\nFrozen features + fold-local scaling"
            "\nRepresentation and regularisation search\nSeed 20261027"
        ),
        48,
        162,
        425,
        165,
        21,
    )
    text(s, "Transfer and preprocessing", 510, 112, 405, 32, 23, bold=True)
    text(
        s,
        (
            "RGB, resize/crop, pretrained normalisation\nTraining only: crops, "
            "flips, mild colour jitter\nResNet: 10 fixed epochs\nDINOv2: bounded "
            "search, 5 final epochs"
        ),
        510,
        162,
        402,
        170,
        20,
    )
    text(
        s,
        "Choose on development data. Lock every model. Evaluate once on the same 630 images.",
        48,
        373,
        864,
        75,
        26,
        bold=True,
    )
    text(
        s,
        "DINOv2 adaptation is exploratory. Test ranking does not choose the primary.",
        48,
        467,
        864,
        37,
        18,
        "A25524",
    )

    s = slide(
        "Actual learning curves",
        50,
        (
            "These are the complete recorded histories for all three development "
            "folds at the selected DINO backbone learning rate of 1e-5. Solid "
            "curves show training and dashed curves development validation. No "
            "curve is a fresh-test result. The training fit becomes strong while "
            "validation varies by source fold. Validation chooses the learning "
            "rate and epoch count, so it is optimistic for that selection. The "
            "six-fit search also includes 3e-6, whose mean best development Macro"
            " F1 was 0.76250 versus 0.77248 at 1e-5. Best epochs 6, 5 and 3 imply"
            " five final epochs. ResNet has actual training-only logs for its "
            "fixed schedule and no epoch-wise validation curve. The frozen "
            "logistic head has regularisation-search diagnostics, not neural "
            "learning epochs."
        ),
        [
            f"reports/results/v3/finetune_development/lr_1e-05_fold_{i}/progress.json"
            for i in range(3)
        ]
        + [
            "reports/results/v3/finetune_development/selection.json",
            "reports/results/v3/resnet_baseline/final_history.csv",
        ],
    )
    histories = [
        read(RAW / f"finetune_development/lr_1e-05_fold_{i}/progress.json")["history"]
        for i in range(3)
    ]
    colors = ["137F82", "A25524", "65549D"]
    for j, metric in enumerate(["loss", "macro_f1"]):
        ss = []
        for i, hist in enumerate(histories):
            for split, key in [("train", "train"), ("validation", "validation_development_only")]:
                ss.append(
                    dict(
                        name=f"Fold {i + 1} {split}",
                        values=[h[key][metric] for h in hist],
                        color=colors[i],
                        dash=split == "validation",
                    )
                )
        chart(
            s,
            "Loss" if j == 0 else "Macro F1",
            list(range(1, max(map(len, histories)) + 1)),
            ss,
            40 + j * 455,
            125,
            440,
            300,
            ymax=1 if j else None,
            ylabel="Cross-entropy" if j == 0 else "Macro F1",
        )
    text(
        s,
        "DINOv2, selected learning rate 1e-5. Solid: training. Dashed: development validation.",
        48,
        443,
        864,
        32,
        18,
    )
    text(
        s,
        "Training fit improves faster than validation. Fresh test data never set epochs.",
        48,
        484,
        864,
        28,
        19,
        bold=True,
    )

    s = slide(
        "Clean test results on unseen sources",
        55,
        (
            "The primary achieves 505 correct images out of 630, or 80.16%, with "
            "Macro F1 0.7952. Its 95% source-group bootstrap interval is 66.03 to"
            " 93.65%. The point target of 80% is met but the lower bound is below"
            " 80%. Fine-tuned DINOv2 records 86.83% and Macro F1 0.8631. It stays"
            " an exploratory comparison. Its paired accuracy advantage over the "
            "primary is 6.67 percentage points with interval minus 2.06 to plus "
            "16.19, which includes zero. Thus the ranking does not establish "
            "superiority. The control interval is 61.59 to 88.73%, ResNet 63.65 "
            "to 90.48%, and DINOv2 76.98 to 95.87%. Intervals use 10,000 "
            "class-stratified cluster draws over 21 sources, conditional on these"
            " saved predictions."
        ),
        [f"reports/results/v3/fresh/{m}/evaluation.json" for m in models]
        + ["reports/results/v3/report_inputs/paired_fresh.csv"],
    )
    data = [["Pipeline", "Accuracy", "Macro P", "Macro R", "Macro F1"]]
    for m, n in zip(models, names):
        z = metrics[m]
        data.append(
            [
                n,
                f"{z['accuracy'] * 100:.2f}%",
                f"{z['macro_precision']:.3f}",
                f"{z['macro_recall']:.3f}",
                f"{z['macro_f1']:.4f}",
            ]
        )
    table(s, data, 48, 115, 864, 228, [0.36, 0.16, 0.16, 0.16, 0.16], 20)
    text(s, "Primary: 80.16% accuracy, 95% CI 66.03-93.65%", 48, 368, 864, 40, 26, bold=True)
    text(
        s,
        "The 80% point target is met. The lower confidence bound is below 80%.",
        48,
        416,
        864,
        38,
        20,
    )
    text(
        s,
        (
            "Exploratory DINOv2 advantage: +6.67 pp, paired 95% CI -2.06 to "
            "+16.19 pp.\nHigher observed accuracy does not establish superiority."
        ),
        48,
        463,
        864,
        52,
        18,
        "A25524",
    )

    s = slide(
        "Primary errors and per-class performance",
        55,
        (
            "Read the matrix by rows: true class first, prediction second. Every "
            "row has 90 images. Ceramic is often called stone, with 30 of its 90 "
            "images following that error. Leather is called concrete in 22 cases."
            " Wood, metal and fabric reach perfect recall on the three sampled "
            "sources per class, which is not a population guarantee. Ceramic "
            "precision is 1.000 but recall is only 0.578, illustrating that "
            "missing many examples can coexist with few false positive labels. "
            "All values are saved point metrics, not newly scored results."
        ),
        [
            "reports/results/v3/fresh/frozen_selected/image_confusion_matrix.csv",
            "reports/results/v3/fresh/frozen_selected/image_per_class.csv",
        ],
    )
    cm = rows(RAW / "fresh/frozen_selected/image_confusion_matrix.csv")
    short = ["Wd", "Mt", "St", "Co", "Ce", "Fa", "Le"]
    table(
        s,
        [["True / pred."] + short]
        + [[r["true_class"].title()] + [r[c] for c in classes] for r in cm],
        42,
        150,
        520,
        275,
        [0.24] + [0.76 / 7] * 7,
        16,
        True,
    )
    pc = rows(RAW / "fresh/frozen_selected/image_per_class.csv")
    table(
        s,
        [["Class", "P", "R", "F1"]]
        + [
            [r["class"].title()] + [f"{float(r[k]):.3f}" for k in ["precision", "recall", "f1"]]
            for r in pc
        ],
        585,
        150,
        333,
        275,
        [0.34, 0.22, 0.22, 0.22],
        16,
    )
    text(s, "Confusion counts (90 per true class)", 44, 112, 520, 28, 19, bold=True)
    text(s, "Precision, recall and F1", 585, 112, 333, 28, 19, bold=True)
    text(
        s,
        "Wd wood   Mt metal   St stone   Co concrete   Ce ceramic   Fa fabric   Le leather",
        44,
        438,
        870,
        28,
        15,
    )
    text(
        s,
        "Ceramic to stone: 30 errors. Leather to concrete: 22 errors.",
        48,
        480,
        860,
        34,
        24,
        bold=True,
    )

    s = slide(
        "Correct and incorrect predictions",
        40,
        (
            "These four examples use the saved primary predictions. The correct "
            "wood and metal examples are from the unfiltered class gallery. The "
            "ceramic and leather failures come from the report error gallery. "
            "They are selected illustrations rather than a representative "
            "performance estimate. Point to visual ambiguity in texture and "
            "reflectance as a plausible explanation, not a causal attribution. "
            "Model probabilities can be high for wrong answers and are not "
            "calibrated certainty. Show the predicted class and confidence for "
            "each."
        ),
        [
            "reports/results/v3/report_inputs/gallery_selection.json",
            "reports/results/v3/fresh/frozen_selected/predictions.csv",
        ],
    )
    for i, (r, p) in enumerate(zip(examples, ex_paths)):
        x = 48 + i * 224
        pred = classes[int(r["y_pred"])]
        truth = classes[int(r["y_true"])]
        image(s, p, x, 145, 196, 196)
        text(s, "True: " + truth, x, 359, 205, 28, 19, bold=True)
        text(s, "Predicted: " + pred, x, 392, 205, 28, 18)
        text(
            s,
            f"Score: {float(r['p_' + pred]) * 100:.1f}%",
            x,
            424,
            205,
            28,
            18,
            "137F82" if truth == pred else "A25524",
        )
    text(
        s,
        "A high predicted probability can accompany an incorrect label.",
        48,
        482,
        860,
        35,
        23,
        bold=True,
    )

    s = slide(
        "JPEG compression exposes a weakness",
        40,
        (
            "These saved corruption results use the same 630 images and five "
            "deterministic transforms. The primary drops from 80.16% clean "
            "accuracy to 59.05% at JPEG quality 30, a 21.11 percentage point "
            "loss. DINOv2 falls from 86.83% to 75.40%. Primary corruption tests "
            "were predeclared, whereas the DINOv2 corruption comparison was "
            "applied after observing clean and primary-corruption scores and is "
            "explicitly post hoc. Compression is the largest degradation here, "
            "but these tests do not substitute for new scenes or real-camera "
            "data."
        ),
        [
            "reports/results/v3/robustness/primary/comparison.csv",
            "reports/results/v3/robustness/dinov2_exploratory/comparison.csv",
            "reports/results/v3/exploratory_robustness_amendment.json",
        ],
    )
    conditions = ["Clean", "Darker", "Brighter", "Blur", "JPEG 30", "128 px"]
    robust = [
        rows(RAW / f"robustness/{folder}/comparison.csv")
        for folder in ["primary", "dinov2_exploratory"]
    ]
    series = [
        dict(
            name=name,
            values=[metrics[m]["accuracy"] * 100] + [float(r["accuracy"]) * 100 for r in rr],
            color=c,
        )
        for name, m, rr, c in zip(
            ["Primary", "DINOv2 (post hoc)"],
            ["frozen_selected", "dinov2_finetuned"],
            robust,
            ["137F82", "A25524"],
        )
    ]
    chart(
        s,
        "Accuracy under image corruption",
        conditions,
        series,
        45,
        110,
        870,
        320,
        kind="bar",
        ymax=100,
        ylabel="Accuracy (%)",
    )
    text(s, "Primary: 80.16% clean to 59.05% at JPEG quality 30", 48, 443, 865, 38, 27, bold=True)
    text(
        s,
        "DINOv2 corruption results are a post hoc exploratory comparison.",
        48,
        489,
        865,
        29,
        18,
        "A25524",
    )

    s = slide(
        "Efficiency and model recommendation",
        45,
        (
            "On the recorded RTX 4070 Laptop GPU, ResNet has the lowest warm "
            "batch-one median latency. The primary is slower because it uses "
            "multiple backbone passes. Required storage includes the backbones, "
            "not just the small classifier head. Timing starts from predecoded "
            "RGB and includes preprocessing and probabilities, but excludes disk "
            "reading and decoding. These are CUDA measurements, not classroom CPU"
            " measurements. Keep frozen fusion as the preselected study result. "
            "Fine-tuned DINOv2 is a promising next deployment candidate because "
            "it has higher observed accuracy and lower measured latency than the "
            "primary, but it needs an independently planned confirmation. Choose "
            "ResNet only if the speed trade-off is acceptable."
        ),
        [f"reports/results/v3/benchmarks/{m}/benchmark.json" for m in models],
    )
    data = [["Pipeline", "Warm median", "Storage"]]
    for m, n in zip(models, names):
        b = read(RAW / f"benchmarks/{m}/benchmark.json")
        one = next(x for x in b["results"] if x["scope"] == "warm" and x["batch_size"] == 1)
        data.append(
            [
                n,
                f"{one['median_seconds'] * 1000:.2f} ms",
                f"{b['deployment']['total_deployment_bytes'] / 1e6:.1f} MB",
            ]
        )
    table(s, data, 48, 113, 864, 215, [0.50, 0.25, 0.25], 21)
    text(
        s,
        (
            "Retain frozen fusion as the primary study model.\nConfirm DINOv2 on "
            "new sources before a deployment decision."
        ),
        48,
        357,
        864,
        80,
        25,
        bold=True,
    )
    text(
        s,
        (
            "RTX 4070 Laptop GPU. Warm batch 1, predecoded RGB.\nIncludes "
            "preprocessing and probabilities, excludes disk I/O and decoding."
        ),
        48,
        461,
        864,
        48,
        17,
    )

    s = slide(
        "Limitations and next experiment",
        40,
        (
            "Only 21 new groups, three per class, give wide uncertainty despite "
            "630 images. Asset provenance is audited but some common physical "
            "ancestry and backbone pretraining overlap may remain unknown. "
            "Sources combine scanned and procedural textures, and restricted "
            "class definitions and renderer differences affect transfer. No "
            "genuine photograph test has been performed. Pipeline comparison also"
            " combines architecture, resolution, pretraining and compute changes."
            " The next experiment should be frozen before data collection, add "
            "more independent sources in ceramic, concrete, stone and leather, "
            "vary scene and camera, include real photographs, and test "
            "compression systematically. Reusing this now-scored holdout for "
            "tuning would require another fresh assessment."
        ),
        [
            "reports/v3/report.md, sections 3, 5 and 16",
            "reports/results/v3/provenance_verification.json",
        ],
    )
    text(s, "Few independent sources", 48, 119, 410, 33, 25, bold=True)
    text(
        s,
        (
            "Three test groups per class\nRelated views share texture evidence\n"
            "Unknown ancestry can remain"
        ),
        48,
        168,
        410,
        115,
        22,
    )
    text(s, "Restricted domain", 510, 119, 402, 33, 25, bold=True)
    text(
        s,
        "Scanned and procedural assets\nControlled Blender scenes\nNo real-photograph validation",
        510,
        168,
        402,
        115,
        22,
    )
    text(
        s,
        (
            "Next: a fixed evaluation plan with more independent sources,\nnew "
            "scenes and photographs, and compression coverage."
        ),
        48,
        337,
        864,
        103,
        27,
        bold=True,
    )
    text(
        s,
        "Model comparisons do not isolate architecture or equal-compute effects.",
        48,
        477,
        864,
        35,
        19,
        "A25524",
    )

    s = slide(
        "Material Studio demo and conclusion",
        50,
        (
            "Use the separate classroom guide. The expected app sequence is "
            "Classify, Results, then Dataset. Select an image or built-in sample "
            "and model, start a prediction, let the loading state complete, and "
            "inspect probabilities. Only call this a live prediction after the "
            "actual runtime has completed inference for that input. In Results, "
            "compare stored model metrics and confusions, then inspect the class "
            "gallery in Dataset. The app is under parallel integration and this "
            "workflow is not yet verified here. If runtime inference is "
            "unavailable, explicitly announce the offline fallback and use stored"
            " results or slide 9 examples. Conclude that source-aware evaluation "
            "supports the primary point target, but uncertainty and compression "
            "weakness limit broad claims."
        ),
        ["docs/CLASSROOM_DEMO.md", "reports/results/v3/fresh/frozen_selected/evaluation.json"],
    )
    text(s, "Classify", 48, 118, 245, 36, 27, bold=True)
    text(s, "Image or sample\nChoose model\nLoading, then probabilities", 48, 168, 260, 121, 23)
    text(s, "Results", 350, 118, 245, 36, 27, bold=True)
    text(
        s, "Stored model comparison\nConfusion matrices\nClass-level errors", 350, 168, 260, 121, 23
    )
    text(s, "Dataset", 652, 118, 260, 36, 27, bold=True)
    text(
        s,
        "Browse the gallery\nInspect source variety\nExplain group separation",
        652,
        168,
        260,
        121,
        23,
    )
    text(
        s,
        (
            "The primary meets the clean point target.\nWide uncertainty and JPEG "
            "weakness limit the claim."
        ),
        48,
        339,
        864,
        88,
        29,
        bold=True,
    )
    text(
        s,
        "Offline fallback: show stored results, clearly labelled as stored.",
        48,
        471,
        864,
        36,
        20,
        "A25524",
    )

    s = slide(
        "Sources",
        10,
        (
            "Experimental claims come from the saved evaluations, prediction "
            "tables, training histories and audit records cited in each slide "
            "note. These papers explain the model families and evaluation "
            "principles. Provider pages establish the source-texture licence, not"
            " a licence decision for the rendered collection. All presenters "
            "should be able to discuss the full study. Invite questions."
        ),
        [
            "ITRI626 practical project brief, sections 10 and 12.2",
            "reports/v3/report.md, References",
        ],
    )
    refs = [
        ("Oquab et al. (2024), DINOv2", "arxiv.org/abs/2304.07193"),
        ("Darcet et al. (2024), Vision Transformers Need Registers", "arxiv.org/abs/2309.16588"),
        ("Liu et al. (2022), A ConvNet for the 2020s", "arxiv.org/abs/2201.03545"),
        ("He et al. (2016), Deep Residual Learning", "arxiv.org/abs/1512.03385"),
        ("Cawley & Talbot (2010), Model selection bias", "jmlr.org/papers/v11/cawley10a.html"),
        (
            "Poly Haven and ambientCG, CC0 asset licences",
            "polyhaven.com/license    docs.ambientcg.com/license/",
        ),
    ]
    for i, (a, b) in enumerate(refs):
        text(s, a, 48, 112 + i * 54, 865, 25, 18, bold=True)
        text(s, b, 48, 137 + i * 54, 865, 22, 15, "137F82")
    text(
        s,
        (
            "Study evidence: saved evaluations, predictions, training histories "
            "and source audits.\nExact local sources are cited in slide notes and "
            "the speaking guide."
        ),
        48,
        458,
        864,
        55,
        18,
    )
    assert len(slides) == 14 and sum(s["seconds"] for s in slides) == 600
    write(
        OUT / "deck.json",
        json.dumps(dict(slide_width=960, slide_height=540, font="Aptos", slides=slides), indent=2)
        + "\n",
    )
    guide = [
        "# Speaking guide",
        "",
        (
            "A 14-slide rehearsal plan of approximately 10 minutes. The lecturer "
            "has not assigned a duration in the supplied brief. Adapt the pace "
            "when it is announced. Group-member fields are intentionally blank. "
            "All members must participate and be ready to answer questions about "
            "the full project."
        ),
        "",
        (
            "Suggested unnamed hand-offs: slides 1-5, slides 6-10, slides 11-14. "
            "Adjust these for equal speaking time and actual group size."
        ),
        "",
    ]
    for i, s in enumerate(slides, 1):
        guide += (
            [
                f"## {i}. {s['title'].replace(chr(10), ' ')} ({s['seconds']} seconds)",
                "",
                s["notes"],
                "",
                "Sources:",
            ]
            + ["- " + p for p in s["sources"]]
            + [""]
        )
    guide += [
        "## Questions to rehearse",
        "",
        (
            "- Why group by source? Multiple views of one texture share "
            "information and cannot be treated as independent samples."
        ),
        (
            "- Why retain the primary at 80.16%? Development evidence determined "
            "the model before the fresh test. Picking from the test ranking would"
            " change the meaning of that assessment."
        ),
        (
            "- Is DINOv2 better? Its observed accuracy is higher, but the paired "
            "interval includes zero and the follow-up is exploratory."
        ),
        (
            "- Why does an 80% target coexist with a 66.03% lower bound? A point "
            "target is not a guaranteed population lower bound."
        ),
        (
            "- Is the dataset entirely scanned? No. The provider collection mixes"
            " scanned and procedural sources and known relatives are grouped."
        ),
        (
            "- What does model probability mean? A model score for the available "
            "classes, not a validated guarantee that a prediction is correct."
        ),
        (
            "- Why can validation curves look optimistic? They were used to "
            "select learning rate and epoch count."
        ),
        "- Can it classify real photos? This study has not validated that use.",
        (
            "- Who publishes the dataset? The user retains the Kaggle upload and "
            "rendered-dataset licensing decision. No upload is performed by this "
            "presentation workflow."
        ),
        "",
    ]
    write(OUT / "SPEAKING_GUIDE.md", "\n".join(guide))
    write(
        OUT / "qa/evidence_checks.json",
        json.dumps(
            dict(
                slides=14,
                rehearsal_seconds=600,
                raw_scores_verified=True,
                holdout_rows_each=630,
                holdout_groups_each=21,
                example_labels_verified=True,
                images=assets,
                model_training_or_inference_performed=False,
            ),
            indent=2,
        )
        + "\n",
    )
    if not args.prepare_only:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path(__file__).with_suffix(".ps1")),
                "-DeckPath",
                str(OUT / "deck.json"),
            ],
            check=True,
        )
        add_native_charts(slides)
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path(__file__).with_suffix(".ps1")),
                "-DeckPath",
                str(OUT / "deck.json"),
                "-RenderOnly",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

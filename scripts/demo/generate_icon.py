"""Generate Material Studio's cube SVG, PNGs and multi-resolution Windows icons.

Run with the project's existing Pillow dependency: python scripts/demo/generate_icon.py.
Geometry and flat neutral colors are shared by every output; no external fonts,
renderers, logo assets or network access are needed. --preview writes a size sheet.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
VERTICES = ((128, 20), (222, 74), (222, 182), (128, 236), (34, 182), (34, 74))
CENTER = (128, 128)
FACES = (
    ("#e1e2e3", (VERTICES[0], VERTICES[1], CENTER, VERTICES[5])),
    ("#b4b6b9", (VERTICES[5], CENTER, VERTICES[3], VERTICES[4])),
    ("#85898e", (CENTER, VERTICES[1], VERTICES[2], VERTICES[3])),
)
STROKE = "#45494e"
STROKE_WIDTH = 6


def svg():
    polygons = "\n".join(
        f'  <polygon fill="{color}" points="' + " ".join(f"{x},{y}" for x, y in points) + '"/>'
        for color, points in FACES
    )
    outline = "M " + " L ".join(f"{x},{y}" for x, y in VERTICES) + " Z"
    edges = " ".join(f"M 128,128 L {x},{y}" for x, y in (VERTICES[1], VERTICES[3], VERTICES[5]))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" '
        'width="256" height="256" role="img" aria-labelledby="title">\n'
        '  <title id="title">Material Studio cube</title>\n'
        f'{polygons}\n  <path d="{outline} {edges}" fill="none" stroke="{STROKE}" '
        f'stroke-width="{STROKE_WIDTH}" stroke-linejoin="round" stroke-linecap="round"/>\n'
        "</svg>\n"
    )


def raster(size):
    # Supersample each requested size directly so small icons retain clean edges.
    scale = size * 4 / 256
    image = Image.new("RGBA", (size * 4, size * 4))
    draw = ImageDraw.Draw(image)

    def points(values):
        return [(x * scale, y * scale) for x, y in values]

    for color, face in FACES:
        draw.polygon(points(face), fill=color)
    width = max(1, round(STROKE_WIDTH * scale))
    draw.line(points((*VERTICES, VERTICES[0])), fill=STROKE, width=width)
    for vertex in (VERTICES[1], VERTICES[3], VERTICES[5]):
        draw.line(points((CENTER, vertex)), fill=STROKE, width=width)
    # Round caps/joins match the SVG's simple linework.
    radius = STROKE_WIDTH * scale / 2
    for x, y in points((*VERTICES, CENTER)):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=STROKE)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def generate(output, preview=None):
    output.mkdir(parents=True, exist_ok=True)
    vector = svg()
    for name in ("icon.svg", "favicon.svg"):
        (output / name).write_text(vector, encoding="utf-8", newline="\n")
    images = {size: raster(size) for size in SIZES}
    for size, image in images.items():
        image.save(output / f"icon-{size}.png")
    for name, sizes in (("icon.ico", SIZES), ("favicon.ico", (16, 32, 48))):
        images[max(sizes)].save(
            output / name,
            format="ICO",
            sizes=[(s, s) for s in sizes],
            append_images=[images[s] for s in sizes if s != max(sizes)],
        )
    if preview:
        sheet = Image.new("RGB", (660, 660), "#f2f2f2")
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 330, 660, 660), fill="#242629")
        for row in range(2):
            x = 16
            for size in (16, 32, 48, 64, 128, 256):
                sheet.paste(images[size], (x, row * 330 + 36), images[size])
                draw.text((x, row * 330 + 308), str(size), fill="#888888")
                x += size + 16
        preview.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[2] / "app/assets"
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    generate(args.output, args.preview)

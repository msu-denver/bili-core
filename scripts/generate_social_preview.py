"""Generate the GitHub social preview image for bili-core.

Produces ``docs/img/social-preview.png`` at 1280x640 (GitHub's recommended
OpenGraph card dimensions). The maintainer uploads the rendered PNG via
Settings -> General -> Social preview; the source is committed so the
image can be re-rendered if branding evolves.

Composition: the bili-core logo on the left, project name and three-
component tagline on the right, on a dark background that matches the
existing logo's backdrop.

Run with any Python that has Pillow installed:

    python scripts/generate_social_preview.py

The script tries Avenir Next, Helvetica Neue, Helvetica, then DejaVu Sans,
then Pillow's built-in default. macOS will land on Avenir Next; Linux will
land on DejaVu Sans. Output is pixel-stable for a given Pillow + font combo
(the PNG byte stream may shift across zlib/Pillow patch releases under
``optimize=True``, but the rendered pixels are deterministic).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = REPO_ROOT / "bili" / "images" / "logo.png"
OUTPUT_PATH = REPO_ROOT / "docs" / "img" / "social-preview.png"

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 640

BACKGROUND_COLOR = (48, 58, 68, 255)  # Matches logo's flat-fill backdrop
TEXT_COLOR_PRIMARY = (255, 255, 255, 255)
TEXT_COLOR_SECONDARY = (165, 180, 188, 255)  # Soft cool grey
TEXT_COLOR_ACCENT = (85, 191, 239, 255)  # Teal accent from the logo

LOGO_SIZE = 480  # Square logo box on the left
LEFT_PADDING = 64  # Distance from canvas edge to logo
GAP_BETWEEN_LOGO_AND_TEXT = 56  # Space between logo right edge and text block

TITLE_SIZE = 108
TAGLINE_SIZE = 38
COMPONENTS_SIZE = 36
INSTITUTION_SIZE = 26

TITLE_TEXT = "BiliCore"
TAGLINE_TEXT = "Open-Source LLM Framework"
COMPONENTS_TEXT = "IRIS  ·  AETHER  ·  AEGIS"
INSTITUTION_TEXT = "C3 Lab  ·  MSU Denver"

# Search paths for clean sans-serif fonts. First match wins. The script falls
# back to Pillow's bitmap default if none of these resolve.
FONT_SEARCH_PATHS_REGULAR = [
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
FONT_SEARCH_PATHS_BOLD = [
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


# For .ttc font collections, the face-index layout varies per collection.
# These maps record the bold and regular face indexes for each known
# collection. Unknown collections in the search paths are skipped rather
# than silently falling through to face 0, which would degrade a bold
# request to whatever weight that collection happens to put at index 0
# (typically Regular).
TTC_BOLD_FACE_INDEX = {
    "Avenir Next.ttc": 0,  # Avenir Next Bold
    "HelveticaNeue.ttc": 1,  # Helvetica Neue Bold
    "Helvetica.ttc": 1,  # Helvetica Bold
}
TTC_REGULAR_FACE_INDEX = {
    "Avenir Next.ttc": 7,  # Avenir Next Regular
    "HelveticaNeue.ttc": 0,  # Helvetica Neue Regular
    "Helvetica.ttc": 0,  # Helvetica Regular
}


def load_font(
    paths: list[str], size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the first font from ``paths`` that loads. Falls back to default.

    For .ttc collections, looks up the correct bold or regular face index in
    the per-collection maps above. Unknown .ttc collections are skipped so a
    bold request never silently degrades to a regular face.
    """
    index_map = TTC_BOLD_FACE_INDEX if bold else TTC_REGULAR_FACE_INDEX
    for path in paths:
        try:
            if path.endswith(".ttc"):
                collection_name = next(
                    (name for name in index_map if name in path), None
                )
                if collection_name is None:
                    continue
                return ImageFont.truetype(
                    path, size=size, index=index_map[collection_name]
                )
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default(size=size)


def main() -> None:
    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND_COLOR)

    # ----- Logo -----
    logo = Image.open(LOGO_PATH).convert("RGBA")
    logo = logo.resize((LOGO_SIZE, LOGO_SIZE), Image.LANCZOS)
    logo_x = LEFT_PADDING
    logo_y = (CANVAS_HEIGHT - LOGO_SIZE) // 2
    canvas.paste(logo, (logo_x, logo_y), logo)

    # ----- Text block -----
    draw = ImageDraw.Draw(canvas)
    text_x = logo_x + LOGO_SIZE + GAP_BETWEEN_LOGO_AND_TEXT

    title_font = load_font(FONT_SEARCH_PATHS_BOLD, TITLE_SIZE, bold=True)
    tagline_font = load_font(FONT_SEARCH_PATHS_REGULAR, TAGLINE_SIZE)
    components_font = load_font(FONT_SEARCH_PATHS_BOLD, COMPONENTS_SIZE, bold=True)
    institution_font = load_font(FONT_SEARCH_PATHS_REGULAR, INSTITUTION_SIZE)

    # Measure all four text rows so we can vertically center the block.
    def text_height(text: str, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[3] - bbox[1]

    title_h = text_height(TITLE_TEXT, title_font)
    tagline_h = text_height(TAGLINE_TEXT, tagline_font)
    components_h = text_height(COMPONENTS_TEXT, components_font)
    institution_h = text_height(INSTITUTION_TEXT, institution_font)

    spacing_after_title = 36
    spacing_after_tagline = 40
    spacing_after_components = 28
    total_block_h = (
        title_h
        + spacing_after_title
        + tagline_h
        + spacing_after_tagline
        + components_h
        + spacing_after_components
        + institution_h
    )

    cursor_y = (CANVAS_HEIGHT - total_block_h) // 2

    draw.text((text_x, cursor_y), TITLE_TEXT, fill=TEXT_COLOR_PRIMARY, font=title_font)
    cursor_y += title_h + spacing_after_title

    draw.text(
        (text_x, cursor_y), TAGLINE_TEXT, fill=TEXT_COLOR_SECONDARY, font=tagline_font
    )
    cursor_y += tagline_h + spacing_after_tagline

    draw.text(
        (text_x, cursor_y),
        COMPONENTS_TEXT,
        fill=TEXT_COLOR_ACCENT,
        font=components_font,
    )
    cursor_y += components_h + spacing_after_components

    draw.text(
        (text_x, cursor_y),
        INSTITUTION_TEXT,
        fill=TEXT_COLOR_SECONDARY,
        font=institution_font,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

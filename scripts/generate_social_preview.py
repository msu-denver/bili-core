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

The script tries Avenir Next, Helvetica, then DejaVu Sans, then Pillow's
built-in default. macOS will land on Avenir Next; Linux will land on
DejaVu Sans. Output is byte-stable for a given Pillow + font combo.
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


def load_font(paths: list[str], size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Return the first font from ``paths`` that loads. Falls back to default.

    For .ttc collections (macOS Avenir Next.ttc), face 0 is Bold and face 7 is
    Regular. Other .ttc collections use face 0 for the primary face and may
    bundle a Bold via the same collection at varying indexes; falling back to
    index 0 there gets the right face for the regular case and an acceptable
    weight for the bold case.
    """
    for path in paths:
        try:
            if "Avenir Next.ttc" in path:
                index = 0 if bold else 7
                return ImageFont.truetype(path, size=size, index=index)
            if path.endswith(".ttc"):
                return ImageFont.truetype(path, size=size, index=0)
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

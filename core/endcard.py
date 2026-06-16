"""One-click "thank you for reading" end card for a chapter.

Renders a neat, self-contained last page — heading, scanlation name and a
Discord call-to-action — as a clean image that drops in as the final page of a
release. Pure Pillow, no models or network, so it's instant."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

# Display face for the big headline (comic feel); body uses a clean bold sans.
_DISPLAY_FONTS = [
    os.path.join(_FONT_DIR, "Bangers-Regular.ttf"),
    os.path.join(_FONT_DIR, "ComicNeue-Bold.ttf"),
]
_BODY_FONTS = [
    os.path.join(_FONT_DIR, "ComicNeue-Bold.ttf"),
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

# A couple of tasteful themes. Colors are RGB.
THEMES = {
    "dark":  {"bg": (20, 22, 28),    "panel": (28, 31, 39),  "ink": (236, 238, 245),
              "muted": (150, 156, 170), "accent": (88, 101, 242), "accent_ink": (255, 255, 255),
              "rule": (60, 65, 80)},
    "light": {"bg": (247, 248, 250), "panel": (255, 255, 255), "ink": (24, 26, 32),
              "muted": (120, 126, 140), "accent": (88, 101, 242), "accent_ink": (255, 255, 255),
              "rule": (214, 218, 228)},
}


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _font(paths, size):
    path = _first_existing(paths)
    try:
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _fit_font(paths, text, max_w, start, min_size=14):
    """Largest font from `paths` at which `text` fits within `max_w`."""
    size = start
    while size > min_size:
        f = _font(paths, size)
        if _text_w(f, text) <= max_w:
            return f
        size -= 2
    return _font(paths, min_size)


def _text_w(font, text):
    box = font.getbbox(text)
    return box[2] - box[0]


def _draw_center(draw, cx, y, text, font, fill, tracking=0):
    """Draw `text` horizontally centered on `cx` at top-`y`. Optional letter
    tracking (extra px between glyphs) for the small kicker line."""
    if tracking <= 0:
        w = _text_w(font, text)
        draw.text((cx - w / 2, y), text, font=font, fill=fill)
        return
    widths = [_text_w(font, ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


def make_end_card(
    scanlation: str = "BorutoTBV Scanlations",
    discord: str = "discord.gg/borutotbv",
    width: int = 1200,
    height: int = 1700,
    theme: str = "dark",
    heading: str = "THANK YOU FOR READING",
    kicker: str = "END OF CHAPTER",
    footer: str = "Please support the official release",
) -> np.ndarray:
    """Render the end card and return it as a BGR numpy image (cv2-ready)."""
    width = max(600, int(width))
    height = max(800, int(height))
    t = THEMES.get((theme or "dark").lower(), THEMES["dark"])
    scanlation = (scanlation or "").strip() or "Scanlations"
    discord = (discord or "").strip()

    img = Image.new("RGB", (width, height), t["bg"])
    d = ImageDraw.Draw(img)

    # Inset double-rule frame — gives it a finished, deliberate look.
    m = int(min(width, height) * 0.05)
    d.rectangle([m, m, width - m, height - m], outline=t["rule"], width=3)
    d.rectangle([m + 10, m + 10, width - m - 10, height - m - 10], outline=t["rule"], width=1)

    cx = width / 2
    inner_w = width - 2 * m - 80
    cy = height * 0.30   # start the centered stack a bit above the middle

    # Kicker (small, letter-spaced, muted)
    kf = _font(_BODY_FONTS, int(height * 0.020))
    _draw_center(d, cx, cy, kicker.upper(), kf, t["muted"], tracking=int(height * 0.006))
    cy += int(height * 0.020) + int(height * 0.03)

    # Heading — the big line. Wrap to two lines if it doesn't fit on one.
    hsize = int(height * 0.075)
    words = heading.upper().split()
    hf = _fit_font(_DISPLAY_FONTS, heading.upper(), inner_w, hsize)
    lines = [heading.upper()]
    if _text_w(hf, heading.upper()) > inner_w and len(words) > 1:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        hf = _fit_font(_DISPLAY_FONTS, max(lines, key=len), inner_w, hsize)
    lh = (hf.getbbox("Ag")[3] - hf.getbbox("Ag")[1]) + int(height * 0.012)
    for ln in lines:
        _draw_center(d, cx, cy, ln, hf, t["ink"])
        cy += lh
    cy += int(height * 0.02)

    # Short accent rule under the heading
    rw = int(inner_w * 0.18)
    d.rectangle([cx - rw / 2, cy, cx + rw / 2, cy + 5], fill=t["accent"])
    cy += int(height * 0.045)

    # Scanlation name (accent colored, prominent)
    nf = _fit_font(_BODY_FONTS, scanlation, inner_w, int(height * 0.040))
    _draw_center(d, cx, cy, scanlation, nf, t["accent"])
    cy += (nf.getbbox("Ag")[3] - nf.getbbox("Ag")[1]) + int(height * 0.07)

    # Discord call-to-action pill
    if discord:
        label = "JOIN OUR DISCORD"
        lf = _font(_BODY_FONTS, int(height * 0.0235))
        df = _fit_font(_BODY_FONTS, discord, inner_w * 0.8, int(height * 0.028))
        pad_x, pad_y = int(width * 0.05), int(height * 0.022)
        gap = int(height * 0.012)
        lw = _text_w(lf, label)
        dw = _text_w(df, discord)
        lhh = lf.getbbox("Ag")[3] - lf.getbbox("Ag")[1]
        dhh = df.getbbox("Ag")[3] - df.getbbox("Ag")[1]
        pill_w = max(lw, dw) + 2 * pad_x
        pill_h = lhh + dhh + gap + 2 * pad_y
        x0, y0 = cx - pill_w / 2, cy
        d.rounded_rectangle([x0, y0, x0 + pill_w, y0 + pill_h],
                            radius=int(pill_h * 0.28), fill=t["accent"])
        _draw_center(d, cx, y0 + pad_y - lf.getbbox(label)[1], label, lf,
                     t["accent_ink"], tracking=int(height * 0.003))
        _draw_center(d, cx, y0 + pad_y + lhh + gap - df.getbbox(discord)[1],
                     discord, df, t["accent_ink"])
        cy = y0 + pill_h

    # Footer, pinned near the bottom rule
    if footer:
        ff = _fit_font(_BODY_FONTS, footer, inner_w, int(height * 0.018))
        _draw_center(d, cx, height - m - int(height * 0.055), footer, ff, t["muted"])

    rgb = np.array(img)
    return rgb[:, :, ::-1].copy()   # RGB -> BGR for cv2.imwrite

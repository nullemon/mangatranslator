"""One-click "thank you for reading" end card for a chapter.

Renders a neat, self-contained last page — heading, scanlation name and a
Discord call-to-action — in several polished styles (ornate frame, manga
halftone, neon glow, minimal, ribbon). Pure Pillow, no models or network."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

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

DISCORD = (88, 101, 242)   # Discord blurple

# Each style is a self-contained look. `base`/`base2` give a vertical gradient
# (equal = flat). Colors are RGB.
STYLES = {
    "ornate": {
        "label": "Ornate (dark)",
        "base": (18, 20, 27), "base2": (12, 13, 18),
        "ink": (238, 240, 248), "muted": (150, 156, 172),
        "accent": (214, 178, 95), "accent_ink": (20, 18, 12),
        "rule": (214, 178, 95), "frame": "ornate",
        "pill": (214, 178, 95), "pill_ink": (24, 20, 10),
    },
    "halftone": {
        "label": "Manga halftone (light)",
        "base": (246, 242, 233), "base2": (246, 242, 233),
        "ink": (22, 22, 26), "muted": (110, 110, 118),
        "accent": (225, 60, 70), "accent_ink": (255, 255, 255),
        "rule": (20, 20, 24), "frame": "manga", "halftone": True,
        "pill": (20, 20, 24), "pill_ink": (255, 255, 255),
        "heading_shadow": (225, 60, 70),
    },
    "neon": {
        "label": "Neon glow (dark)",
        "base": (10, 12, 24), "base2": (18, 22, 44),
        "ink": (236, 240, 255), "muted": (132, 140, 175),
        "accent": (94, 234, 212), "accent_ink": (6, 14, 14),
        "rule": (94, 234, 212), "frame": "neon", "glow": True,
        "pill": (94, 234, 212), "pill_ink": (6, 16, 16),
    },
    "minimal": {
        "label": "Minimal (light)",
        "base": (251, 251, 253), "base2": (251, 251, 253),
        "ink": (26, 28, 34), "muted": (140, 145, 158),
        "accent": DISCORD, "accent_ink": (255, 255, 255),
        "rule": (222, 225, 233), "frame": "hairline",
        "pill": DISCORD, "pill_ink": (255, 255, 255),
    },
    "blurple": {
        "label": "Discord (dark)",
        "base": (20, 22, 28), "base2": (20, 22, 28),
        "ink": (236, 238, 245), "muted": (150, 156, 170),
        "accent": DISCORD, "accent_ink": (255, 255, 255),
        "rule": (60, 65, 80), "frame": "double",
        "pill": DISCORD, "pill_ink": (255, 255, 255),
    },
}

# Back-compat for the old dark/light "theme" values.
_THEME_ALIAS = {"dark": "blurple", "light": "minimal"}


# ─────────────────────────────── helpers ────────────────────────────────
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


def _text_w(font, text):
    box = font.getbbox(text)
    return box[2] - box[0]


def _text_h(font, text="Ag"):
    box = font.getbbox(text)
    return box[3] - box[1]


def _fit_font(paths, text, max_w, start, min_size=14):
    size = start
    while size > min_size:
        f = _font(paths, size)
        if _text_w(f, text) <= max_w:
            return f
        size -= 2
    return _font(paths, min_size)


def _vgrad(w, h, top, bottom):
    if tuple(top) == tuple(bottom):
        return Image.new("RGB", (w, h), tuple(top))
    t = np.linspace(0, 1, h)[:, None]
    col = np.array(top, float) * (1 - t) + np.array(bottom, float) * t
    arr = np.repeat(col[:, None, :], w, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _diamond(d, cx, cy, r, fill):
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill)


def _halftone_layer(w, h, color, spacing, max_r):
    """Screentone dots that fade in from the four corners — the classic manga
    background feel without overwhelming the text in the middle."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = w / 2, h / 2
    maxd = (cx ** 2 + cy ** 2) ** 0.5
    for yy in range(0, h + spacing, spacing):
        for xx in range(0, w + spacing, spacing):
            dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
            f = max(0.0, (dist / maxd - 0.55) / 0.45)   # 0 center → 1 corners
            r = max_r * f
            if r >= 0.6:
                d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color)
    return layer


def _center(d, cx, y, text, font, fill, tracking=0, stroke=0, stroke_fill=None):
    """Draw text horizontally centered on cx at top-y. Optional letter tracking
    (extra px between glyphs) and an optional outline stroke."""
    if tracking <= 0:
        w = _text_w(font, text)
        d.text((cx - w / 2, y), text, font=font, fill=fill,
               stroke_width=stroke, stroke_fill=stroke_fill)
        return
    widths = [_text_w(font, ch) or font.size * 0.3 for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for ch, wch in zip(text, widths):
        d.text((x, y), ch, font=font, fill=fill,
               stroke_width=stroke, stroke_fill=stroke_fill)
        x += wch + tracking


def _glow_text(img, cx, y, text, font, fill, glow_color, radius):
    """Centered text with a soft outer glow (rendered on a blurred layer)."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    w = _text_w(font, text)
    ld.text((cx - w / 2, y), text, font=font, fill=glow_color + (255,))
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    img.alpha_composite(layer)
    ImageDraw.Draw(img).text((cx - w / 2, y), text, font=font, fill=fill + (255,))


# ─────────────────────────────── frames ─────────────────────────────────
def _frame(img, d, st, m, W, H):
    kind = st.get("frame")
    rule = st["rule"]
    accent = st["accent"]
    if kind == "double":
        d.rectangle([m, m, W - m, H - m], outline=rule, width=3)
        d.rectangle([m + 10, m + 10, W - m - 10, H - m - 10], outline=rule, width=1)
    elif kind == "hairline":
        d.rectangle([m, m, W - m, H - m], outline=rule, width=2)
    elif kind == "ornate":
        # double frame + corner brackets + corner diamonds — an elegant look
        d.rectangle([m, m, W - m, H - m], outline=rule, width=2)
        inset = m + 14
        d.rectangle([inset, inset, W - inset, H - inset], outline=rule, width=1)
        L = int(min(W, H) * 0.06)
        for (cxp, cyp, sx, sy) in [(m, m, 1, 1), (W - m, m, -1, 1),
                                   (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            d.line([(cxp, cyp), (cxp + sx * L, cyp)], fill=accent, width=5)
            d.line([(cxp, cyp), (cxp, cyp + sy * L)], fill=accent, width=5)
            _diamond(d, cxp + sx * 7, cyp + sy * 7, 6, accent)
        # top & bottom center diamonds
        _diamond(d, W / 2, m, 9, accent)
        _diamond(d, W / 2, H - m, 9, accent)
    elif kind == "manga":
        # bold action-comic frame: thick outer + thin inner, halftone behind
        d.rectangle([m, m, W - m, H - m], outline=rule, width=9)
        d.rectangle([m + 16, m + 16, W - m - 16, H - m - 16], outline=rule, width=2)
    elif kind == "neon":
        # glowing rounded border: blurred stroke layer + crisp line on top
        r = int(min(W, H) * 0.04)
        box = [m, m, W - m, H - m]
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.rounded_rectangle(box, radius=r, outline=accent + (255,), width=6)
        glow = glow.filter(ImageFilter.GaussianBlur(9))
        img.alpha_composite(glow)
        ImageDraw.Draw(img).rounded_rectangle(box, radius=r, outline=accent + (255,), width=3)
        ImageDraw.Draw(img).rounded_rectangle(
            [m + 12, m + 12, W - m - 12, H - m - 12], radius=max(2, r - 12),
            outline=accent + (90,), width=1)


# ─────────────────────────────── text stack ─────────────────────────────
def _draw_stack(img, st, W, H, m, scanlation, discord, heading, kicker, footer):
    rgba = img.mode == "RGBA"
    d = ImageDraw.Draw(img)
    cx = W / 2
    inner_w = W - 2 * m - 80
    accent = st["accent"]
    ink = st["ink"]
    muted = st["muted"]
    glow = st.get("glow")
    cy = H * 0.27

    # Kicker
    kf = _font(_BODY_FONTS, int(H * 0.020))
    _center(d, cx, cy, kicker.upper(), kf, muted, tracking=int(H * 0.006))
    cy += int(H * 0.020) + int(H * 0.032)

    # Heading (wrap to two lines if needed)
    hsize = int(H * 0.078)
    words = heading.upper().split()
    hf = _fit_font(_DISPLAY_FONTS, heading.upper(), inner_w, hsize)
    lines = [heading.upper()]
    if _text_w(hf, heading.upper()) > inner_w and len(words) > 1:
        mid = (len(words) + 1) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        hf = _fit_font(_DISPLAY_FONTS, max(lines, key=len), inner_w, hsize)
    lh = _text_h(hf) + int(H * 0.014)
    for ln in lines:
        if glow and rgba:
            _glow_text(img, cx, cy, ln, hf, ink, accent, int(H * 0.012))
            d = ImageDraw.Draw(img)
        elif st.get("heading_shadow"):
            off = max(2, int(H * 0.004))
            _center(d, cx + off, cy + off, ln, hf, st["heading_shadow"])
            _center(d, cx, cy, ln, hf, ink)
        else:
            _center(d, cx, cy, ln, hf, ink)
        cy += lh
    cy += int(H * 0.018)

    # Accent rule with diamond end-caps
    rw = int(inner_w * 0.20)
    ry = cy + 2
    d.line([(cx - rw / 2, ry), (cx + rw / 2, ry)], fill=accent, width=4)
    _diamond(d, cx - rw / 2, ry, 7, accent)
    _diamond(d, cx + rw / 2, ry, 7, accent)
    cy += int(H * 0.05)

    # Scanlation name
    nf = _fit_font(_BODY_FONTS, scanlation, inner_w, int(H * 0.042))
    if glow and rgba:
        _glow_text(img, cx, cy, scanlation, nf, accent, accent, int(H * 0.008))
        d = ImageDraw.Draw(img)
    else:
        _center(d, cx, cy, scanlation, nf, accent)
    cy += _text_h(nf) + int(H * 0.065)

    # Discord call-to-action pill
    if discord:
        label = "JOIN OUR DISCORD"
        lf = _font(_BODY_FONTS, int(H * 0.0235))
        df = _fit_font(_BODY_FONTS, discord, inner_w * 0.8, int(H * 0.030))
        pad_x, pad_y = int(W * 0.05), int(H * 0.022)
        gap = int(H * 0.010)
        lw, dw = _text_w(lf, label), _text_w(df, discord)
        lhh, dhh = _text_h(lf, label), _text_h(df, discord)
        pill_w = max(lw, dw) + 2 * pad_x
        pill_h = lhh + dhh + gap + 2 * pad_y
        x0, y0 = cx - pill_w / 2, cy
        box = [x0, y0, x0 + pill_w, y0 + pill_h]
        rad = int(pill_h * 0.30)
        if glow and rgba:
            gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(gl).rounded_rectangle(box, radius=rad, fill=accent + (255,))
            img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(12)))
            d = ImageDraw.Draw(img)
        d.rounded_rectangle(box, radius=rad, fill=st["pill"])
        _center(d, cx, y0 + pad_y - lf.getbbox(label)[1], label, lf,
                st["pill_ink"], tracking=int(H * 0.003))
        _center(d, cx, y0 + pad_y + lhh + gap - df.getbbox(discord)[1],
                discord, df, st["pill_ink"])
        cy = y0 + pill_h

    # Footer
    if footer:
        ff = _fit_font(_BODY_FONTS, footer, inner_w, int(H * 0.018))
        _center(d, cx, H - m - int(H * 0.055), footer, ff, muted)


# ─────────────────────────────── public API ─────────────────────────────
def make_end_card(
    scanlation: str = "BorutoTBV Scanlations",
    discord: str = "discord.gg/borutotbv",
    width: int = 1200,
    height: int = 1700,
    style: str = "ornate",
    theme: str = "",
    heading: str = "THANK YOU FOR READING",
    kicker: str = "END OF CHAPTER",
    footer: str = "Please support the official release",
) -> np.ndarray:
    """Render the end card and return it as a BGR numpy image (cv2-ready)."""
    W = max(600, int(width))
    H = max(800, int(height))
    key = (style or "").lower().strip()
    if key in _THEME_ALIAS:                 # old "dark"/"light" callers
        key = _THEME_ALIAS[key]
    if not key and theme:
        key = _THEME_ALIAS.get(theme.lower(), "ornate")
    st = STYLES.get(key, STYLES["ornate"])
    scanlation = (scanlation or "").strip() or "Scanlations"
    discord = (discord or "").strip()

    img = _vgrad(W, H, st["base"], st["base2"])
    if st.get("halftone"):
        img = img.convert("RGBA")
        dots = _halftone_layer(W, H, st["rule"] + (60,),
                               spacing=max(14, W // 48), max_r=max(5, W // 150))
        img.alpha_composite(dots)
    if st.get("glow"):
        img = img.convert("RGBA")

    m = int(min(W, H) * 0.05)
    _frame(img, ImageDraw.Draw(img), st, m, W, H)
    _draw_stack(img, st, W, H, m, scanlation, discord, heading, kicker, footer)

    if img.mode == "RGBA":
        img = img.convert("RGB")
    rgb = np.array(img)
    return rgb[:, :, ::-1].copy()   # RGB -> BGR for cv2.imwrite

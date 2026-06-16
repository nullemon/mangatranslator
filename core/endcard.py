"""One-click "thank you for reading" end card for a chapter.

Renders a self-contained, premium-looking last page — heading, scanlation name
and a Discord call-to-action — in a range of polished styles, including a few
series-inspired themes (color + motif only; no trademarked logos or art).
Pure Pillow, no models or network. A custom accent color can override any style.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")


def _fp(name):
    return os.path.join(_FONT_DIR, name)


# Font families (first existing file wins). Variable fonts carry an optional
# weight to instance. System fallbacks keep things working without the bundle.
_SYS_BOLD = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONTS = {
    "bangers":    [_fp("Bangers-Regular.ttf")],
    "anton":      [_fp("Anton-Regular.ttf")],
    "bebas":      [_fp("BebasNeue-Regular.ttf")],
    "pirata":     [_fp("PirataOne-Regular.ttf")],
    "reggae":     [_fp("ReggaeOne-Regular.ttf")],
    "cinzeldec":  [_fp("CinzelDecorative-Black.ttf"), _fp("CinzelDecorative-Bold.ttf")],
    "cinzel":     [_fp("Cinzel.ttf")],
    "marcellus":  [_fp("Marcellus-Regular.ttf")],
    "orbitron":   [_fp("Orbitron.ttf")],
    "comic":      [_fp("ComicNeue-Bold.ttf")],
    "body":       [_fp("ComicNeue-Bold.ttf")] + _SYS_BOLD,
}
# Default variable-font weight to instance per family.
_VAR_WEIGHT = {"orbitron": 800, "cinzel": 600}

DISCORD = (88, 101, 242)


# ─────────────────────────────── font loading ───────────────────────────
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


_FONT_CACHE = {}


def _font(family, size):
    size = max(6, int(size))
    key = (family, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _first_existing(FONTS.get(family, [])) or _first_existing(_SYS_BOLD)
    try:
        f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    w = _VAR_WEIGHT.get(family)
    if w is not None:
        try:
            f.set_variation_by_axes([w])
        except Exception:
            pass
    _FONT_CACHE[key] = f
    return f


def _text_w(font, text):
    b = font.getbbox(text)
    return b[2] - b[0]


def _text_h(font, text="Ag"):
    b = font.getbbox(text)
    return b[3] - b[1]


def _fit(family, text, max_w, start, min_size=14):
    s = start
    while s > min_size:
        if _text_w(_font(family, s), text) <= max_w:
            return _font(family, s)
        s -= 2
    return _font(family, min_size)


# ─────────────────────────────── canvas helpers ─────────────────────────
def _vgrad(w, h, top, bottom):
    if tuple(top) == tuple(bottom):
        return Image.new("RGB", (w, h), tuple(top))
    t = np.linspace(0, 1, h)[:, None]
    col = np.array(top, float) * (1 - t) + np.array(bottom, float) * t
    return Image.fromarray(np.repeat(col[:, None, :], w, axis=1).astype(np.uint8), "RGB")


def _radial(img, cx, cy, radius, color, alpha=120):
    """Soft radial glow centered at (cx,cy)."""
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              fill=tuple(color) + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(radius * 0.5))
    img.alpha_composite(layer)


def _vignette(img, strength=0.5):
    w, h = img.size
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    mask = np.clip(1 - strength * (d ** 2.2), 0, 1)
    arr = np.array(img.convert("RGB")).astype(np.float32) * mask[..., None]
    out = Image.fromarray(arr.astype(np.uint8), "RGB")
    return out.convert(img.mode)


def _paper(img, grain=14, seed=7):
    """Aged-paper grain for parchment looks."""
    arr = np.array(img.convert("RGB")).astype(np.int16)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-grain, grain, arr.shape[:2])
    arr = np.clip(arr + noise[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert(img.mode)


def _halftone(img, color, spacing, max_r, threshold=0.55):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = w / 2, h / 2
    maxd = math.hypot(cx, cy)
    for yy in range(0, h + spacing, spacing):
        for xx in range(0, w + spacing, spacing):
            f = max(0.0, (math.hypot(xx - cx, yy - cy) / maxd - threshold) / (1 - threshold))
            r = max_r * f
            if r >= 0.6:
                d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color)
    img.alpha_composite(layer)


def _hex_pattern(img, color, size, width=2):
    """Faint hexagon mesh (soccer / tech vibe)."""
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    dx = size * 1.5
    dy = size * math.sqrt(3)
    row = 0
    y = -dy
    while y < h + dy:
        off = 0 if row % 2 == 0 else dx / 2 * 0  # pointy-top columns
        x = -dx
        while x < w + dx:
            pts = [(x + size * math.cos(math.radians(a)),
                    y + size * math.sin(math.radians(a))) for a in range(0, 360, 60)]
            d.polygon(pts, outline=color, width=width)
            x += dx
        y += dy / 2
        row += 1
    img.alpha_composite(layer)


# ─────────────────────────────── motifs ─────────────────────────────────
def _spiral(d, cx, cy, r, color, width=6, turns=2.6):
    pts = []
    steps = int(turns * 80)
    for i in range(steps + 1):
        t = i / steps
        ang = turns * 2 * math.pi * t
        rad = r * t
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    if len(pts) > 1:
        d.line(pts, fill=color, width=width, joint="curve")


def _skull(d, cx, cy, s, color, hole):
    """Tiny jolly-roger emblem."""
    d.ellipse([cx - s, cy - s, cx + s, cy + s * 0.85], fill=color)          # cranium
    d.rectangle([cx - s * 0.55, cy + s * 0.45, cx + s * 0.55, cy + s * 1.05], fill=color)  # jaw
    d.ellipse([cx - s * 0.5, cy - s * 0.35, cx - s * 0.08, cy + s * 0.1], fill=hole)  # eye L
    d.ellipse([cx + s * 0.08, cy - s * 0.35, cx + s * 0.5, cy + s * 0.1], fill=hole)  # eye R
    d.polygon([(cx, cy + s * 0.15), (cx - s * 0.12, cy + s * 0.45),
               (cx + s * 0.12, cy + s * 0.45)], fill=hole)                  # nose
    for off in (-1, 1):                                                     # jaw lines
        d.line([(cx + off * s * 0.2, cy + s * 0.5), (cx + off * s * 0.2, cy + s * 1.0)],
               fill=hole, width=max(1, int(s * 0.06)))


def _crossbones(d, cx, cy, s, color):
    for ang in (35, -35):
        a = math.radians(ang)
        dx, dy = math.cos(a) * s, math.sin(a) * s
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=color, width=max(3, int(s * 0.12)))
        for ex in (-1, 1):
            kx, ky = cx + ex * dx, cy + ex * dy
            d.ellipse([kx - s * 0.14, ky - s * 0.14, kx + s * 0.14, ky + s * 0.14], fill=color)


def _laurel(d, cx, cy, r, color, side=1):
    """A curved laurel branch (half wreath) opening toward `side`."""
    for i in range(7):
        t = i / 6
        ang = math.radians(-70 + 140 * t)
        bx = cx + side * r * 0.1 + side * math.sin(ang) * r * 0.15
        by = cy - math.cos(ang) * r
        lx = bx + side * math.cos(ang) * r * 0.28
        ly = by + math.sin(ang) * r * 0.28
        d.line([(cx, cy), (cx + side * r * 0.12, by)], fill=color, width=3)
        d.ellipse([lx - r * 0.09, ly - r * 0.05, lx + r * 0.09, ly + r * 0.05], fill=color)


def _eye(d, cx, cy, w, color, ink):
    """A sharp stylized eye (Blue Lock-ish)."""
    d.polygon([(cx - w, cy), (cx, cy - w * 0.5), (cx + w, cy), (cx, cy + w * 0.5)],
              outline=color, width=max(2, int(w * 0.06)))
    d.ellipse([cx - w * 0.32, cy - w * 0.32, cx + w * 0.32, cy + w * 0.32], fill=color)
    d.ellipse([cx - w * 0.13, cy - w * 0.13, cx + w * 0.13, cy + w * 0.13], fill=ink)


def _diamond(d, cx, cy, r, fill):
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill)


# ─────────────────────────────── text drawing ───────────────────────────
def _center(d, cx, y, text, font, fill, tracking=0, stroke=0, stroke_fill=None):
    if tracking <= 0:
        w = _text_w(font, text)
        d.text((cx - w / 2, y), text, font=font, fill=fill,
               stroke_width=stroke, stroke_fill=stroke_fill)
        return
    widths = [(_text_w(font, ch) or font.size * 0.3) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for ch, wch in zip(text, widths):
        d.text((x, y), ch, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        x += wch + tracking


def _glow_text(img, cx, y, text, font, fill, glow_color, radius):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    w = _text_w(font, text)
    ImageDraw.Draw(layer).text((cx - w / 2, y), text, font=font, fill=tuple(glow_color) + (255,))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius)))
    ImageDraw.Draw(img).text((cx - w / 2, y), text, font=font, fill=tuple(fill) + (255,))


# ─────────────────────────────── style table ────────────────────────────
# Each style: palette + chosen fonts + a `look` key dispatched for bg/frame/motif.
STYLES = {
    "royal": {
        "label": "Royal (gold)", "look": "royal",
        "base": (16, 14, 22), "base2": (8, 7, 12),
        "ink": (244, 235, 210), "muted": (180, 165, 130),
        "accent": (212, 175, 90), "accent_ink": (24, 18, 8),
        "rule": (212, 175, 90), "pill": (212, 175, 90), "pill_ink": (24, 18, 8),
        "f_head": "cinzeldec", "f_kick": "marcellus", "f_name": "cinzel", "f_body": "marcellus",
        "head_caps": True,
    },
    "naruto": {
        "label": "Ninja orange", "look": "naruto",
        "base": (20, 18, 16), "base2": (8, 7, 6),
        "ink": (250, 240, 228), "muted": (180, 150, 120),
        "accent": (240, 130, 25), "accent_ink": (20, 14, 6),
        "rule": (240, 130, 25), "pill": (240, 130, 25), "pill_ink": (24, 14, 4),
        "f_head": "reggae", "f_kick": "bebas", "f_name": "reggae", "f_body": "body",
        "head_caps": True, "head_shadow": (120, 50, 0),
    },
    "onepiece": {
        "label": "Pirate poster", "look": "onepiece",
        "base": (232, 214, 176), "base2": (214, 190, 146),
        "ink": (40, 28, 16), "muted": (120, 95, 60),
        "accent": (160, 38, 30), "accent_ink": (242, 226, 192),
        "rule": (40, 28, 16), "pill": (40, 28, 16), "pill_ink": (240, 224, 188),
        "f_head": "pirata", "f_kick": "marcellus", "f_name": "pirata", "f_body": "marcellus",
        "head_caps": True,
    },
    "ragnarok": {
        "label": "Gods & gold", "look": "ragnarok",
        "base": (24, 22, 24), "base2": (10, 9, 11),
        "ink": (236, 230, 222), "muted": (160, 150, 140),
        "accent": (201, 162, 39), "accent_ink": (18, 14, 6),
        "rule": (201, 162, 39), "pill": (142, 27, 27), "pill_ink": (244, 232, 210),
        "f_head": "cinzeldec", "f_kick": "cinzel", "f_name": "cinzel", "f_body": "marcellus",
        "head_caps": True, "head_shadow": (110, 20, 20),
    },
    "bluelock": {
        "label": "Striker blue", "look": "bluelock",
        "base": (8, 14, 32), "base2": (16, 30, 70),
        "ink": (232, 240, 255), "muted": (130, 150, 195),
        "accent": (0, 209, 255), "accent_ink": (4, 16, 28),
        "rule": (0, 209, 255), "pill": (0, 160, 255), "pill_ink": (4, 14, 26),
        "f_head": "orbitron", "f_kick": "anton", "f_name": "orbitron", "f_body": "body",
        "head_caps": True, "glow": True,
    },
    "neon": {
        "label": "Neon glow", "look": "neon",
        "base": (10, 12, 24), "base2": (18, 22, 44),
        "ink": (236, 240, 255), "muted": (132, 140, 175),
        "accent": (94, 234, 212), "accent_ink": (6, 14, 14),
        "rule": (94, 234, 212), "pill": (94, 234, 212), "pill_ink": (6, 16, 16),
        "f_head": "anton", "f_kick": "bebas", "f_name": "orbitron", "f_body": "body",
        "head_caps": True, "glow": True,
    },
    "minimal": {
        "label": "Minimal", "look": "minimal",
        "base": (251, 251, 253), "base2": (251, 251, 253),
        "ink": (26, 28, 34), "muted": (140, 145, 158),
        "accent": DISCORD, "accent_ink": (255, 255, 255),
        "rule": (222, 225, 233), "pill": DISCORD, "pill_ink": (255, 255, 255),
        "f_head": "bangers", "f_kick": "body", "f_name": "body", "f_body": "body",
        "head_caps": True,
    },
}
_ALIAS = {"dark": "royal", "light": "minimal", "ornate": "royal", "halftone": "naruto",
          "blurple": "neon"}


def _hex(c):
    if not c:
        return None
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    except Exception:
        return None


# ─────────────────────────────── backgrounds + frames ───────────────────
def _paint_bg(img, st):
    look = st["look"]
    W, H = img.size
    if look == "naruto":
        _radial(img, W * 0.5, H * 0.42, W * 0.5, st["accent"], alpha=46)
        for (cx, cy) in [(W * 0.12, H * 0.12), (W * 0.88, H * 0.88)]:
            _spiral(ImageDraw.Draw(img), cx, cy, W * 0.12,
                    tuple(st["accent"]) + (60,) if img.mode == "RGBA" else st["accent"],
                    width=5)
    elif look == "onepiece":
        # parchment grain + aged vignette + faint center jolly roger
        img.paste(_paper(img, grain=16))
        sk = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sk)
        _crossbones(sd, W / 2, H * 0.5, W * 0.18, (60, 44, 28, 38))
        _skull(sd, W / 2, H * 0.5, W * 0.1, (60, 44, 28, 40), (232, 214, 176, 0))
        img.alpha_composite(sk) if img.mode == "RGBA" else img.paste(sk, (0, 0), sk)
    elif look == "ragnarok":
        # subtle marble veins
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        rng = np.random.default_rng(11)
        for _ in range(14):
            x0 = rng.integers(0, W)
            pts = [(x0, 0)]
            x = x0
            for y in range(0, H, 40):
                x += int(rng.integers(-26, 26))
                pts.append((x, y))
            ld.line(pts, fill=(150, 140, 120, 22), width=2)
        img.alpha_composite(layer)
    elif look == "bluelock":
        _hex_pattern(img, (90, 150, 230, 30), size=46, width=2)
        # diagonal speed shards
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for i in range(-2, 8):
            x = i * W * 0.18
            ld.polygon([(x, 0), (x + W * 0.05, 0), (x - W * 0.18, H), (x - W * 0.23, H)],
                       fill=(0, 180, 255, 14))
        img.alpha_composite(layer)


def _paint_frame(img, st, m):
    look = st["look"]
    W, H = img.size
    d = ImageDraw.Draw(img)
    rule, accent = st["rule"], st["accent"]
    if look == "minimal":
        d.rectangle([m, m, W - m, H - m], outline=rule, width=2)
    elif look == "neon":
        r = int(min(W, H) * 0.04)
        box = [m, m, W - m, H - m]
        gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).rounded_rectangle(box, radius=r, outline=tuple(accent) + (255,), width=6)
        img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(9)))
        ImageDraw.Draw(img).rounded_rectangle(box, radius=r, outline=tuple(accent) + (255,), width=3)
    elif look == "bluelock":
        # sharp cut-corner frame with neon glow
        c = int(min(W, H) * 0.06)
        pts = [(m + c, m), (W - m - c, m), (W - m, m + c), (W - m, H - m - c),
               (W - m - c, H - m), (m + c, H - m), (m, H - m - c), (m, m + c)]
        gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).line(pts + [pts[0]], fill=tuple(accent) + (255,), width=6)
        img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(8)))
        ImageDraw.Draw(img).line(pts + [pts[0]], fill=tuple(accent) + (255,), width=3)
    elif look == "onepiece":
        d.rectangle([m, m, W - m, H - m], outline=rule, width=8)
        d.rectangle([m + 14, m + 14, W - m - 14, H - m - 14], outline=rule, width=2)
        # corner skulls
        for (cx, cy) in [(m + 30, m + 30), (W - m - 30, m + 30),
                         (m + 30, H - m - 30), (W - m - 30, H - m - 30)]:
            _skull(d, cx, cy, 16, rule, st["base"])
    elif look == "naruto":
        d.rectangle([m, m, W - m, H - m], outline=accent, width=7)
        d.rectangle([m + 14, m + 14, W - m - 14, H - m - 14], outline=accent, width=2)
        for (cx, cy) in [(m + 8, m + 8), (W - m - 8, m + 8),
                         (m + 8, H - m - 8), (W - m - 8, H - m - 8)]:
            _spiral(d, cx, cy, 34, accent, width=4, turns=2.2)
    elif look == "ragnarok":
        d.rectangle([m, m, W - m, H - m], outline=accent, width=3)
        d.rectangle([m + 14, m + 14, W - m - 14, H - m - 14], outline=(142, 27, 27), width=1)
        for (cx, cy, s) in [(m + 4, m + 4, 1), (W - m - 4, m + 4, -1)]:
            d.line([(cx, cy), (cx + s * W * 0.07, cy)], fill=accent, width=6)
            d.line([(cx, cy), (cx, cy + H * 0.05)], fill=accent, width=6)
        for cx in (W * 0.5,):
            _laurel(d, cx - 60, H - m - 2, 46, accent, side=1)
            _laurel(d, cx + 60, H - m - 2, 46, accent, side=-1)
    else:  # royal
        d.rectangle([m, m, W - m, H - m], outline=accent, width=2)
        inset = m + 14
        d.rectangle([inset, inset, W - inset, H - inset], outline=accent, width=1)
        L = int(min(W, H) * 0.06)
        for (cxp, cyp, sx, sy) in [(m, m, 1, 1), (W - m, m, -1, 1),
                                   (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            d.line([(cxp, cyp), (cxp + sx * L, cyp)], fill=accent, width=5)
            d.line([(cxp, cyp), (cxp, cyp + sy * L)], fill=accent, width=5)
            _diamond(d, cxp + sx * 7, cyp + sy * 7, 6, accent)
        _diamond(d, W / 2, m, 9, accent)
        _diamond(d, W / 2, H - m, 9, accent)


# ─────────────────────────────── the text stack ─────────────────────────
def _draw_stack(img, st, m, scanlation, discord, heading, kicker, footer):
    rgba = img.mode == "RGBA"
    d = ImageDraw.Draw(img)
    W, H = img.size
    cx = W / 2
    inner_w = W - 2 * m - 80
    accent, ink, muted = st["accent"], st["ink"], st["muted"]
    glow = st.get("glow")
    head_caps = st.get("head_caps", True)
    cy = H * 0.27

    # Kicker
    kf = _font(st["f_kick"], int(H * 0.021))
    _center(d, cx, cy, kicker.upper(), kf, muted, tracking=int(H * 0.006))
    cy += int(H * 0.021) + int(H * 0.034)

    # Heading
    htext = heading.upper() if head_caps else heading
    hsize = int(H * 0.078)
    hf = _fit(st["f_head"], htext, inner_w, hsize)
    lines = [htext]
    if _text_w(hf, htext) > inner_w and len(htext.split()) > 1:
        words = htext.split()
        mid = (len(words) + 1) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        hf = _fit(st["f_head"], max(lines, key=len), inner_w, hsize)
    lh = _text_h(hf, "AgjÉ") + int(H * 0.016)
    for ln in lines:
        if glow and rgba:
            _glow_text(img, cx, cy, ln, hf, ink, accent, int(H * 0.013))
            d = ImageDraw.Draw(img)
        elif st.get("head_shadow"):
            o = max(2, int(H * 0.0045))
            _center(d, cx + o, cy + o, ln, hf, st["head_shadow"])
            _center(d, cx, cy, ln, hf, ink)
        else:
            _center(d, cx, cy, ln, hf, ink)
        cy += lh
    cy += int(H * 0.016)

    # Accent rule with diamond end-caps
    rw = int(inner_w * 0.20)
    ry = cy + 2
    d.line([(cx - rw / 2, ry), (cx + rw / 2, ry)], fill=accent, width=4)
    _diamond(d, cx - rw / 2, ry, 7, accent)
    _diamond(d, cx + rw / 2, ry, 7, accent)
    cy += int(H * 0.05)

    # Scanlation name
    nf = _fit(st["f_name"], scanlation, inner_w, int(H * 0.040))
    if glow and rgba:
        _glow_text(img, cx, cy, scanlation, nf, accent, accent, int(H * 0.008))
        d = ImageDraw.Draw(img)
    else:
        _center(d, cx, cy, scanlation, nf, accent)
    cy += _text_h(nf, scanlation) + int(H * 0.066)

    # Discord pill
    if discord:
        label = "JOIN OUR DISCORD"
        lf = _font(st["f_body"], int(H * 0.0235))
        df = _fit(st["f_body"], discord, inner_w * 0.8, int(H * 0.030))
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
            ImageDraw.Draw(gl).rounded_rectangle(box, radius=rad, fill=tuple(st["pill"]) + (255,))
            img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(12)))
            d = ImageDraw.Draw(img)
        d.rounded_rectangle(box, radius=rad, fill=st["pill"])
        _center(d, cx, y0 + pad_y - lf.getbbox(label)[1], label, lf,
                st["pill_ink"], tracking=int(H * 0.003))
        _center(d, cx, y0 + pad_y + lhh + gap - df.getbbox(discord)[1], discord, df, st["pill_ink"])
        cy = y0 + pill_h

    # Footer
    if footer:
        ff = _fit(st["f_body"], footer, inner_w, int(H * 0.018))
        _center(d, cx, H - m - int(H * 0.052), footer, ff, muted)


# ─────────────────────────────── public API ─────────────────────────────
def make_end_card(
    scanlation: str = "BorutoTBV Scanlations",
    discord: str = "discord.gg/borutotbv",
    width: int = 1200,
    height: int = 1700,
    style: str = "royal",
    theme: str = "",
    accent: str = "",
    heading: str = "THANK YOU FOR READING",
    kicker: str = "END OF CHAPTER",
    footer: str = "Please support the official release",
) -> np.ndarray:
    """Render the end card and return it as a BGR numpy image (cv2-ready).

    `accent` (optional "#RRGGBB") overrides the style's accent / rule / pill.
    """
    W = max(600, int(width))
    H = max(800, int(height))
    key = (style or "").lower().strip()
    key = _ALIAS.get(key, key)
    if key not in STYLES and theme:
        key = _ALIAS.get(theme.lower().strip(), "royal")
    st = dict(STYLES.get(key, STYLES["royal"]))
    scanlation = (scanlation or "").strip() or "Scanlations"
    discord = (discord or "").strip()

    custom = _hex(accent)
    if custom:
        st["accent"] = custom
        st["rule"] = custom
        st["pill"] = custom
        # keep a readable ink on the pill based on luminance
        lum = 0.299 * custom[0] + 0.587 * custom[1] + 0.114 * custom[2]
        st["pill_ink"] = (20, 20, 24) if lum > 150 else (245, 245, 250)
        st["accent_ink"] = st["pill_ink"]

    img = _vgrad(W, H, st["base"], st["base2"])
    need_rgba = st.get("glow") or st["look"] in ("naruto", "onepiece", "bluelock", "ragnarok")
    if need_rgba:
        img = img.convert("RGBA")

    _paint_bg(img, st)
    if st["look"] == "onepiece":
        img = _vignette(img, 0.5)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    _paint_frame(img, st, int(min(W, H) * 0.05))
    _draw_stack(img, st, int(min(W, H) * 0.05), scanlation, discord, heading, kicker, footer)

    if img.mode == "RGBA":
        img = img.convert("RGB")
    rgb = np.array(img)
    return rgb[:, :, ::-1].copy()

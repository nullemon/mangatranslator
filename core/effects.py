"""Post-render page effects.

raw_scan() makes a clean rendered page look like a rough magazine raw.
Two styles:

  "photo" — a phone photo of a physical Jump page (the classic early-leak
            look): washed-out low-contrast ink, cool gray-beige newsprint,
            big soft lighting bands across the page, binding/edge shadows,
            heavy photo grain.
  "scan"  — an aged flatbed scan: yellowed tan paper, print grain, pulp
            mottle, fiber streaks, dust.

Fully deterministic (fixed seed) so re-renders produce identical texture,
and pure CV — the artwork and lettering are never reinterpreted by a model.

strength 0.4 = subtle, 1.0 = typical, 2.0 = beaten-up copy.
"""
import cv2
import numpy as np


def raw_scan(img: np.ndarray, strength: float = 1.0, seed: int = 1234,
             style: str = "photo") -> np.ndarray:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return img
    rng = np.random.default_rng(seed)
    f = float(np.clip(strength, 0.2, 2.2))
    out = img.astype(np.float32) / 255.0

    if style == "photo":
        g = min(f, 1.6)
        # Phone-photo softness.
        out = cv2.GaussianBlur(out, (0, 0), 0.6 * g)
        # Cool gray-beige paper cast.
        paper = np.array([178, 199, 213], np.float32) / 255.0   # BGR, warm beige
        out *= (1.0 - g * 0.9 * (1.0 - paper))[None, None, :]
        # Newsprint under indoor light: whites drop to ~210, ink floats up
        # to ~75 — the washed, low-contrast leak look. Applied AFTER the
        # tint so the black point lands where we aim it.
        lo = 0.29 * g
        hi = 1.0 - 0.045 * g
        out = lo + out * (hi - lo)
        # Big soft lighting bands: a photographed page is never lit evenly.
        band = rng.standard_normal((7, 5)).astype(np.float32)
        band = cv2.resize(band, (w, h), interpolation=cv2.INTER_CUBIC)
        band = cv2.GaussianBlur(band, (0, 0), min(h, w) / 5.0)
        out *= (1.0 + 0.13 * g * band)[..., None]
        # Binding shadow along the right edge, soft curl shade on top.
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        edge = (np.clip((xx - 0.90) / 0.10, 0, 1) ** 1.5) * 0.30
        edge = edge + (np.clip((0.06 - yy) / 0.06, 0, 1) ** 1.5) * 0.18
        out *= (1.0 - g * edge)[..., None]
        # Photo grain: hard sensor noise + halftone shimmer.
        hard = rng.standard_normal((h, w)).astype(np.float32)
        fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                                (0, 0), 0.8)
        out += (hard * 0.022 + fine * 0.040)[..., None] * g
        # Mild pulp mottle.
        mot = rng.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
        mot = cv2.GaussianBlur(mot, (0, 0), 3.0)
        mot = cv2.resize(mot, (w, h), interpolation=cv2.INTER_LINEAR)
        out += (mot * 0.022)[..., None] * g
        return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

    # ---- style == "scan": aged flatbed scan ----
    # Cheap magazine ink: never razor sharp, never solid black.
    out = cv2.GaussianBlur(out, (0, 0), 0.55 * f)
    out = out * (1.0 - 0.06 * f) + 0.055 * f

    # Yellowed uncoated paper by per-channel multiply: white -> paper.
    paper = np.array([166, 196, 216], np.float32) / 255.0   # BGR, aged tan
    tint = 1.0 - f * 0.85 * (1.0 - paper)
    out *= tint[None, None, :]

    # Print grain: hard per-pixel noise + slightly blurred film grain.
    hard = rng.standard_normal((h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                            (0, 0), 0.9)
    out += (hard * 0.020 + fine * 0.050)[..., None] * f

    # Blotchy paper mottle (uneven pulp) + low-frequency exposure drift.
    mot = rng.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    mot = cv2.GaussianBlur(mot, (0, 0), 3.5)
    mot = cv2.resize(mot, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse = rng.standard_normal((h // 4 + 1, w // 4 + 1)).astype(np.float32)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 24)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    out += (mot * 0.040 + coarse * 0.060)[..., None] * f

    # Paper fibers: faint elongated horizontal streaks in the pulp.
    fib = rng.standard_normal((h, w)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=7.0, sigmaY=0.5)
    out += (fib * 0.030)[..., None] * f

    # Vignette — deeper toward the corners, like a lazy flatbed scan.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx / max(w - 1, 1) - 0.5) * 2.0
    dy = (yy / max(h - 1, 1) - 0.5) * 2.0
    r2 = dx * dx + dy * dy
    out *= (1.0 - 0.09 * f * r2)[..., None]

    # Dust and flecks: dark specks pressed into the paper, a few light ones.
    spk = np.zeros((h, w), np.float32)
    n = int(40 + 140 * f * f * (h * w) / 1_000_000)
    for _ in range(n):
        cv2.circle(spk, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(1, 3)), float(rng.uniform(0.3, 1.0)), -1)
    spk = cv2.GaussianBlur(spk, (0, 0), 0.6)
    out *= (1.0 - 0.5 * f * spk)[..., None]
    lite = np.zeros((h, w), np.float32)
    for _ in range(n // 3):
        cv2.circle(lite, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(1, 2)), float(rng.uniform(0.4, 1.0)), -1)
    out += (cv2.GaussianBlur(lite, (0, 0), 0.5) * 0.10 * f)[..., None]

    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def restore_scan(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Turn a photographed or rough scan back into a clean digital-looking page.

    The opposite journey to raw_scan(). A page shot off a magazine, or ripped
    from a low-grade scan, carries four separate faults at once, and fixing
    only the obvious one leaves it looking grubby:

      the paper is beige or grey rather than white
      the light drifts across the sheet, with a shadow down the binding
      grain and moire sit on top of the artwork
      the ink never reaches black and the paper never reaches white

    They come off in that order, and the order is the whole trick. Stretching
    the levels FIRST — which is all the existing clean-scan finish can do — pulls
    the bright side of an unevenly lit page to white while the shadowed side
    stays muddy, so on a measured test it made the drift across the paper WORSE,
    from 70 levels to 98. Flattening the light first and stretching afterwards
    takes it to 11.

    Two things here are less obvious than they look:

    The lift is gated on how light a pixel already is. The background estimate
    cannot tell a shadow from a large area of solid black ink, so ungated it
    reads a black panel as the darkest shadow on the page and dutifully lifts
    it to mid-grey — measured at 156 out of 255, which is a ruined page. Ink is
    left alone and only paper is lifted.

    The black point is put back by subtracting the floor the photograph added,
    rather than by curve-fitting. A photograph lifts black to around 74 and
    that is simply where the ink now sits; subtracting it is the exact inverse
    and it restores the screentone contrast along with the blacks.

    Pure OpenCV: instant, free, deterministic, and the artwork is never
    reinterpreted the way a generative model would reinterpret it.
    """
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return img
    f = float(np.clip(strength, 0.2, 2.0))
    work = img.astype(np.float32)

    # 1. Paper back to neutral. Per channel, the near-brightest value IS the
    #    paper, whatever colour the photograph made it, so scaling each channel
    #    to put that on white lifts the cast without touching the ink.
    for c in range(3):
        ref = float(np.percentile(work[:, :, c], 97.5))
        if ref > 20:
            work[:, :, c] *= min(3.0, 255.0 / ref)
    work = np.clip(work, 0, 255)

    # 2. Estimate the light falling on the page. Dilating first means the
    #    estimate is built from PAPER, not from whatever art happens to be
    #    underneath. Then down to a thumbnail and back: a Gaussian wide enough
    #    to ignore the artwork also reaches across the binding shadow and
    #    averages it away, leaving the very edge dark — the thumbnail follows
    #    an edge ramp instead of smearing it (drift 46 vs 6 on the same page).
    gray = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    k = max(3, (min(h, w) // 60) | 1)
    paper = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    small_edge = max(8, int(min(h, w) / 12))
    sw = max(4, w * small_edge // min(h, w))
    sh = max(4, h * small_edge // min(h, w))
    small = cv2.medianBlur(cv2.resize(paper, (sw, sh),
                                      interpolation=cv2.INTER_AREA), 3)
    paper = cv2.resize(small.astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    paper = np.maximum(paper, 40.0)

    # 3. Even the light out — on the paper only. `lift` is 0 over ink and 1
    #    over paper, which is what stops a black panel being read as shadow.
    gain = np.clip(float(np.percentile(paper, 95)) / paper, 0.6, 2.0)
    lift = np.clip((gray.astype(np.float32) - 60.0) / 100.0, 0.0, 1.0)
    work = np.clip(work * (1.0 + (gain - 1.0) * lift * f)[..., None], 0, 255)

    # 4. Take the photograph's raised floor back off. Capped, so a page with
    #    no true black on it (a light sketch, an all-screentone panel) is not
    #    forced to invent some.
    g2 = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lo = min(float(np.percentile(g2, 0.5)), 110.0)
    hi = float(np.percentile(g2, 99.5))
    if hi - lo > 40:
        work = np.clip((work - lo) * (255.0 / (hi - lo)), 0, 255)

    # 5. Grain and moire off. Manga is monochrome, so a page that is already
    #    near-grey goes all the way to grey — that removes the last of the cast
    #    and the colour speckle a phone sensor leaves. A page with real colour
    #    in it stays in colour.
    u8 = work.astype(np.uint8)
    if float(cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 26:
        g = cv2.fastNlMeansDenoising(cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY),
                                     None, int(6 * f), 7, 21)
        u8 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    else:
        u8 = cv2.fastNlMeansDenoisingColored(u8, None, int(6 * f), int(6 * f),
                                             7, 21)

    # 6. Put the edge back. A photograph is always slightly soft and the steps
    #    above soften it further; a restrained unsharp returns the line weight
    #    without the white halo a heavy one leaves along the ink.
    blur = cv2.GaussianBlur(u8, (0, 0), 1.1)
    u8 = cv2.addWeighted(u8, 1.0 + 0.55 * f, blur, -0.55 * f, 0)

    # 7. Snap the last of it: the existing finish anchors its curves to this
    #    page's own histogram peaks, so midtones stay put.
    from .pipeline import scan_finish
    return scan_finish(u8)

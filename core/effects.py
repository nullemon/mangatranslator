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

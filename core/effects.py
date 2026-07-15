"""Post-render page effects.

raw_scan() makes a clean rendered page look like a rough weekly-magazine
raw: warm uncoated tan paper, fiber grain, soft print, uneven exposure,
vignette and dust. Fully deterministic (fixed seed) so re-renders of the
same page produce the identical texture, and pure CV — the artwork and
lettering are never reinterpreted by a model.
"""
import cv2
import numpy as np


def raw_scan(img: np.ndarray, strength: float = 1.0, seed: int = 1234) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return img
    rng = np.random.default_rng(seed)
    f = float(np.clip(strength, 0.2, 1.5))
    out = img.astype(np.float32) / 255.0

    # Magazine ink is never razor sharp.
    out = cv2.GaussianBlur(out, (0, 0), 0.55 * f)

    # Warm tan paper by per-channel multiply: white -> paper, ink stays ink.
    paper = np.array([176, 196, 214], np.float32) / 255.0   # BGR, uncoated tan
    tint = 1.0 - f * 0.85 * (1.0 - paper)
    out *= tint[None, None, :]

    # Paper fiber: fine grain plus low-frequency exposure unevenness.
    fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                            (0, 0), 1.1)
    coarse = rng.standard_normal((h // 4 + 1, w // 4 + 1)).astype(np.float32)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 24)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    out += (fine * 0.028 + coarse * 0.045)[..., None] * f

    # Vignette — deeper toward the corners, like a lazy flatbed scan.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx / max(w - 1, 1) - 0.5) * 2.0
    dy = (yy / max(h - 1, 1) - 0.5) * 2.0
    r2 = dx * dx + dy * dy
    out *= (1.0 - 0.07 * f * r2)[..., None]

    # Dust: sparse dark specks pressed into the paper.
    spk = np.zeros((h, w), np.float32)
    n = int(20 + 60 * f * (h * w) / 1_000_000)
    for _ in range(n):
        cv2.circle(spk, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(1, 3)), float(rng.uniform(0.3, 1.0)), -1)
    spk = cv2.GaussianBlur(spk, (0, 0), 0.6)
    out *= (1.0 - 0.5 * f * spk)[..., None]

    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

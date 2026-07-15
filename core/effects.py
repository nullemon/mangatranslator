"""Post-render page effects.

raw_scan() makes a clean rendered page look like a rough weekly-magazine
raw: yellowed uncoated paper, loud print grain, blotchy mottle, fiber
streaks, ink that is never quite black, vignette and dust. Fully
deterministic (fixed seed) so re-renders of the same page produce the
identical texture, and pure CV — the artwork and lettering are never
reinterpreted by a model.

strength 0.4 = lightly aged, 1.0 = typical rough magazine raw,
2.0 = beaten-up newsstand copy.
"""
import cv2
import numpy as np


def raw_scan(img: np.ndarray, strength: float = 1.0, seed: int = 1234) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return img
    rng = np.random.default_rng(seed)
    f = float(np.clip(strength, 0.2, 2.2))
    out = img.astype(np.float32) / 255.0

    # Cheap magazine ink: never razor sharp, never solid black.
    out = cv2.GaussianBlur(out, (0, 0), 0.55 * f)
    out = out * (1.0 - 0.06 * f) + 0.055 * f   # lift blacks toward newsprint ink

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

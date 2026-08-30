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
    out = scan_finish(u8)

    return _flatten_dark_fields(out)


def _flatten_dark_fields(out):
    """The restore's final step, shared with the clean lab.

    A page whose background is dark (night scenes, whole dark-aesthetic
    chapters) keeps its pulp mottle through the paper-oriented steps, and the
    levels stretch then AMPLIFIES what is left — so this runs after the
    stretch, never before it (measured: flattening first left std 11.3 against
    the digital target's 3.8).

    A large, dark, detail-free area is treated as one surface: each connected
    field is filled with its single median tone (a local blur removes speckle
    but leaves the large-scale unevenness that actually reads as blotchy), and
    the pale glow the sharpening pass leaves along panel borders is absorbed
    into the field in two passes. Gated on LOW DETAIL throughout — det < 45 by
    measurement, the glow ramp reading ~30 and true dark hatching ~95 — so
    line art, screentone and hatching are untouched. A page with no large flat
    dark area is returned as it came.
    """
    g8 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    dm = cv2.blur(g8.astype(np.float32), (5, 5))
    dv = cv2.blur(g8.astype(np.float32) ** 2, (5, 5)) - dm ** 2
    det = cv2.blur(np.sqrt(np.maximum(dv, 0)), (9, 9))
    flat_dark = ((g8 < 120) & (det < 16)).astype(np.uint8) * 255
    flat_dark = cv2.morphologyEx(flat_dark, cv2.MORPH_OPEN,
                                 np.ones((9, 9), np.uint8))
    if cv2.countNonZero(flat_dark) > 0.02 * g8.size:
        # One tone per field, not a local blur. A local median removes the
        # speckle but leaves the LARGE-scale unevenness — different parts of
        # the background crushed to different levels by the stretch — and that
        # is what actually reads as blotchy (measured: local detail 0.1 yet
        # the field's spread still 11.8 against the digital page's 3.8). Each
        # connected dark field is a single surface, so it gets its single
        # median colour, feathered at the edges. Separate fields (the page
        # background, a dark tent, a shadow) each keep their own level.
        # And the GLOW along the panel borders. The unsharp pass overshoots
        # where a dark field meets a white panel, and the stretch amplifies
        # that into a pale 20px fringe hanging in the dark — measured, it was
        # the entire remaining unevenness (5,853 of 241,000 background pixels,
        # all in one band beside the border). Pull the ring around each field
        # down to the field's own tone; gated on low detail and on staying
        # below paper level, so the border line itself and any true artwork
        # beside the field are untouched.
        # Absorbed in two passes because the field's own edge sits well back
        # from the border (its ramp fails the flatness gate), so one ring from
        # it does not span the whole glow.
        #
        # det < 45 is from measurement rather than instinct: the glow itself
        # is a steep ramp and reads at det ~30 — a gate of 20 excluded the
        # very thing being removed — while genuine dark hatching beside a
        # field reads at ~95. Bright balloon interiors that also score low
        # are already excluded by the value test.
        for _pass in range(2):
            ring = cv2.dilate(flat_dark, np.ones((41, 41), np.uint8))
            ring[flat_dark > 0] = 0
            halo = (ring > 0) & (g8 < 140) & (det < 45)
            if not halo.any():
                break
            flat_dark[halo] = 255
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(flat_dark, 8)
        fill = out.copy()
        for i in range(1, n2):
            if st2[i, cv2.CC_STAT_AREA] < 0.005 * g8.size:
                continue
            sel = lab2 == i
            fill[sel] = np.median(out[sel].reshape(-1, 3), axis=0)
        wgt = cv2.GaussianBlur(flat_dark.astype(np.float32) / 255.0,
                               (0, 0), 4)[..., None] * 0.9
        out = np.clip(out.astype(np.float32) * (1 - wgt)
                      + fill.astype(np.float32) * wgt, 0, 255).astype(np.uint8)
    return out


# ── the clean lab: ten recipes, one page, you pick ──────────────────────────
#
# Nobody can tune a clean-up from an argument about it — the page has to be
# looked at. Each recipe below moves ONE lever away from the default, so when
# a particular version looks right, that lever is the thing to keep. The
# labels are burnt into the corner of each output so "number 7 is the best"
# is all the feedback needed.

def _restore_tuned(img, *, flatten=1.0, black_cut=0.5, denoise=6.0,
                   sharpen=0.55, snap=True, dark_flatten=True,
                   clahe=False, gamma=1.0, ink_snap=0):
    """restore_scan with its levers exposed. The default arguments reproduce
    restore_scan exactly; the lab varies one at a time."""
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return img
    work = img.astype(np.float32)

    for c in range(3):                                    # white balance
        ref = float(np.percentile(work[:, :, c], 97.5))
        if ref > 20:
            work[:, :, c] *= min(3.0, 255.0 / ref)
    work = np.clip(work, 0, 255)

    if flatten > 0:                                       # even the light
        gray = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        k = max(3, (min(h, w) // 60) | 1)
        paper = cv2.dilate(gray, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)))
        se = max(8, int(min(h, w) / 12))
        sw_, sh_ = max(4, w * se // min(h, w)), max(4, h * se // min(h, w))
        small = cv2.medianBlur(cv2.resize(paper, (sw_, sh_),
                                          interpolation=cv2.INTER_AREA), 3)
        paper = np.maximum(cv2.resize(small.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_CUBIC), 40.0)
        gain = np.clip(float(np.percentile(paper, 95)) / paper, 0.6, 2.0)
        lift = np.clip((gray.astype(np.float32) - 60.0) / 100.0, 0.0, 1.0)
        work = np.clip(work * (1.0 + (gain - 1.0) * lift * flatten)[..., None],
                       0, 255)

    if black_cut > 0:                                     # take the floor off
        g2 = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lo = min(float(np.percentile(g2, black_cut)), 110.0)
        hi = float(np.percentile(g2, 99.5))
        if hi - lo > 40:
            work = np.clip((work - lo) * (255.0 / (hi - lo)), 0, 255)

    u8 = work.astype(np.uint8)
    if gamma != 1.0:
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0,
                      0, 255).astype(np.uint8)
        u8 = cv2.LUT(u8, lut)
    if clahe:
        lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (12, 12)).apply(lab[:, :, 0])
        u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise > 0:                                       # grain off
        if float(cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 26:
            g = cv2.fastNlMeansDenoising(
                cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY), None, int(denoise), 7, 21)
            u8 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        else:
            u8 = cv2.fastNlMeansDenoisingColored(
                u8, None, int(denoise), int(denoise), 7, 21)

    if sharpen > 0:                                       # edge back
        blur = cv2.GaussianBlur(u8, (0, 0), 1.1)
        u8 = cv2.addWeighted(u8, 1.0 + sharpen, blur, -sharpen, 0)

    if snap:                                              # levels snap
        from .pipeline import scan_finish
        u8 = scan_finish(u8)
    if dark_flatten:
        u8 = _flatten_dark_fields(u8)
    if ink_snap > 0:
        # Absolute ink. A digital release's blacks are genuinely 0, not dark
        # grey — it is the single clearest difference between a clean rip and
        # a cleaned scan. Everything at or below the threshold becomes pure
        # black, edges feathered by one blurred step so line ends do not
        # alias. The threshold is the whole judgement call — push it high and
        # mid-grey shading goes with the ink — which is exactly why it is a
        # numbered lab version rather than a default.
        g = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        t = float(ink_snap)
        # 1 at ink, 0 above the threshold band, smooth in between.
        wgt = np.clip((t - g) / 18.0 + 1.0, 0.0, 1.0)
        wgt = cv2.GaussianBlur(wgt, (0, 0), 0.8)[..., None]
        u8 = np.clip(u8.astype(np.float32) * (1.0 - wgt), 0, 255).astype(np.uint8)
    return u8


#: (number, name, what it changes, kwargs). One lever each, both directions.
CLEAN_VARIANTS = [
    (1, "standard", "the current default", {}),
    (2, "gentle", "no levels snap - tones kept soft",
        dict(snap=False, sharpen=0.3)),
    (3, "deep blacks", "harder black floor",
        dict(black_cut=2.0)),
    (4, "soft blacks", "barely touches the floor",
        dict(black_cut=0.1)),
    (5, "flat + smooth", "double denoise",
        dict(denoise=12)),
    (6, "crisp", "double sharpening",
        dict(sharpen=1.1)),
    (7, "no halo", "no sharpening at all",
        dict(sharpen=0.0)),
    (8, "raw fields", "dark backgrounds left un-flattened",
        dict(dark_flatten=False)),
    (9, "punchy", "local contrast (CLAHE) before the snap",
        dict(clahe=True)),
    (10, "ink-heavy", "gamma towards ink",
        dict(gamma=1.25)),
    # The two the user's reference pages actually look like: digital-release
    # blacks, genuinely 0.
    (11, "absolute black", "everything near-ink snapped to pure black",
        dict(ink_snap=70)),
    (12, "pitch black", "ink snap pushed hard - shading may go with it",
        dict(ink_snap=110, gamma=1.1)),
]


def clean_variants(img, max_edge=2200):
    """All ten recipes on one page, labels burnt in.

    Run at a capped size so ten passes stay under half a minute; the winner is
    then re-run at full size (and through HD) by the normal path, so nothing
    about the comparison run limits the final page.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_edge:
        s = max_edge / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    out = []
    for num, name, desc, kw in CLEAN_VARIANTS:
        v = _restore_tuned(img, **kw)
        v = v.copy()
        tag = f"{num}. {name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.rectangle(v, (0, 0), (tw + 26, th + 26), (255, 255, 255), -1)
        cv2.rectangle(v, (0, 0), (tw + 26, th + 26), (0, 0, 0), 2)
        cv2.putText(v, tag, (13, th + 13), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 200), 3)
        out.append((num, name, desc, v))
    return out


def contact_sheet(variants, cols=4):
    """The ten versions on one sheet, for picking at a glance."""
    th, tw = variants[0][3].shape[:2]
    ch, cw = 560, max(1, int(560 * tw / th))
    rows = (len(variants) + cols - 1) // cols
    sheet = np.full((rows * (ch + 8) + 8, cols * (cw + 8) + 8, 3), 30, np.uint8)
    for i, (_n, _name, _d, v) in enumerate(variants):
        r, c = divmod(i, cols)
        y, x = 8 + r * (ch + 8), 8 + c * (cw + 8)
        sheet[y:y + ch, x:x + cw] = cv2.resize(v, (cw, ch),
                                               interpolation=cv2.INTER_AREA)
    return sheet


def clean_page_nokey(img):
    """The recipe behind the Clean — no key workflow.

    This is lab version 12, "pitch black" — chosen by the user's eye against
    eleven alternatives on their own scans, which is the only judgement that
    counts for a look. Ink snapped hard to true 0 with gamma leaning the same
    way: the digital-release look, where black is actually black. If taste
    changes, run the Clean Lab again and point this at the new number.
    """
    return _restore_tuned(img, ink_snap=110, gamma=1.1)

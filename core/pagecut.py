"""Cut the page out of a photograph.

A phone photo of a magazine brings the carpet with it. Everything downstream —
balloon detection, OCR, the erase masks, the AI scan — then spends its effort
on the floor, and the trim and watermark tools are working on a page whose
edges are wherever the photo happened to stop.

This finds the paper and throws the rest away: the background is filled with
white and the image is cropped to the sheet. It is a CROP, so the pixels that
survive are the originals, untouched — nothing is resampled and no quality is
lost.

How the page is found, and how I got it wrong first
---------------------------------------------------
The obvious rule is "paper is bright, carpet is not". That fails on exactly
the photos people take: a beige carpet under warm light sits at 200 and an
off-white page under the same light at 215, and no threshold separates those.

My second guess was that paper is SMOOTH and carpet is textured. Measuring it
showed the opposite, and by a wide margin: on a real-looking photo the page's
local contrast came out at 23.0 against the carpet's 4.4. Of course it did —
a printed page is full of hard black lines, while carpet photographed slightly
out of focus is a soft blur of fibres. The page is the BUSY region, not the
smooth one.

So the page is found as the largest connected area of fine detail, and its
convex hull is the sheet. A hull rather than a rectangle because a page
photographed at an angle is not axis-aligned, and rather than the raw outline
because the margins carry no print and would otherwise be cut off.

Measured against composited photos where the true page is known exactly, over
five carpet tones and angles: IoU 0.96, about 1% of the page lost (its
outermost blank margin) and 4% of the carpet kept. GrabCut was tried as an
alternative and scored 0.99 on a dark carpet but 0.49 on a pale one — the very
case this exists for — as well as being four times slower.

The third lesson came from real photos rather than composited ones. A phone
focuses on the floor as happily as on the page, and a deep-pile carpet in
focus is every bit as busy as print: measured on the user's own photos, the
carpet's local contrast was 20–26 against the page's 22 — a dead tie, and the
detail blob covered the whole frame. What DID separate them, cleanly, was
colour: newsprint is neutral grey (saturation 8–14 there) while a carpet is
beige or brown or blue — something (19–41). So when the detail cue comes back
saying "everything is page", which is its way of failing, the page is re-found
as the largest connected NEUTRAL region instead. Chroma is only trusted as a
fallback because it fails in its own way — a genuinely grey carpet — and on
exactly those photos the carpet blurs softer than print, which is the case the
detail cue already wins.

And the fourth lesson is that hand-built cues have a ceiling. The chroma hull
keeps the artwork perfectly but pays for it with a wedge of carpet at every
corner, because a convex hull cannot follow a bowed page edge. Tracing the
actual boundary is a segmentation problem, and a small local model (U2-Net,
via rembg — the standard background remover) solves it outright: on the same
photos it hugs the sheet's true outline, spine dip and curled corners
included, in about a third of a second on CPU. So when rembg is installed its
mask is used first, cleaned up with the same page-sized/full-frame sanity
rules as everything else, and the classical cues remain as the fallback for
an install without it. Nothing here imports rembg at startup — the first
cut-out pays the model load, and an install without the package never does.
"""
import cv2
import numpy as np

#: analysis size. The page only has to be located to within a pixel or two,
#: and this keeps a 12-megapixel phone photo to a fraction of a second.
WORK = 900

#: how much of the frame must look like a page before anything is cut. Below
#: this, hand the photo back untouched rather than crop it to something that
#: is not the page.
MIN_PAGE = 0.15


def _detail(gray):
    """Local contrast: large over print, small over an out-of-focus carpet.
    Box filters, so it stays fast whatever the size."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (7, 7))
    var = np.maximum(cv2.blur(g * g, (7, 7)) - mean * mean, 0)
    return cv2.blur(np.sqrt(var), (9, 9))


#: a hull covering this much of the frame means the cue did not actually
#: separate page from floor — it just said "everything".
FULL_FRAME = 0.92

#: the U2-Net session, made once and kept. None = not tried yet;
#: False = rembg is not installed (or broke), so don't try again.
_NN = None


def _neural_keep(small: np.ndarray):
    """The sheet as U2-Net sees it, or None when the model cannot help.

    The mask is trusted for its BOUNDARY — that is what the hand-built cues
    cannot do — but it still passes the same sanity rules as every other cue:
    interior holes are page (a colour panel or a dark splash is still the
    page), only page-sized bodies count, and a mask that is nearly the whole
    frame means there is no background to remove.
    """
    global _NN
    if _NN is False:
        return None
    if _NN is None:
        try:
            from rembg import new_session, remove
            _NN = (new_session("u2net"), remove)
        except Exception:
            _NN = False
            print("[pagecut] for a far better page cut-out: "
                  "pip install rembg onnxruntime")
            return None
    sess, remove = _NN
    try:
        m = remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                   session=sess, only_mask=True)
    except Exception:
        return None
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    sh, sw = small.shape[:2]
    if m.shape[:2] != (sh, sw):
        m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_LINEAR)
    keep = (m > 127).astype(np.uint8) * 255
    # Holes inside the sheet are the sheet.
    flood = cv2.copyMakeBorder(keep, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ffm = np.zeros((sh + 4, sw + 4), np.uint8)
    cv2.floodFill(flood, ffm, (0, 0), 255)
    keep = keep | cv2.bitwise_not(flood[1:-1, 1:-1])
    # Specks are not pages; every page-sized body is (a spread can split).
    o = max(3, int(min(sh, sw) * 0.01)) | 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (o, o)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    big = [i for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= MIN_PAGE * sh * sw]
    if not big:
        return None
    keep = (np.isin(lab, big).astype(np.uint8)) * 255
    # A hair of margin, for the model's tendency to sit exactly on the edge.
    g = max(3, int(min(sh, sw) * 0.008)) | 1
    keep = cv2.dilate(keep,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    if int(cv2.countNonZero(keep)) > FULL_FRAME * sh * sw:
        return None
    return keep


def _hull_of(cand: np.ndarray):
    """The sheet as the convex hull of the largest body in a candidate mask.

    Returns a small-size mask, or None when the candidate holds nothing
    page-sized. The hull rather than a rectangle because a page photographed
    at an angle is not axis-aligned; the hull rather than the raw outline
    because the blank margins would otherwise be cut off.
    """
    sh, sw = cand.shape[:2]
    # Close the body into one piece. The gaps being bridged are panel gutters
    # and balloons, which are page-sized, not pixel-sized.
    r = max(5, int(min(sh, sw) * 0.05)) | 1
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < MIN_PAGE * sh * sw:
        return None
    body = (lab == idx).astype(np.uint8)
    cnts, _hier = cv2.findContours(body, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(np.vstack(cnts))
    out = np.zeros((sh, sw), np.uint8)
    cv2.drawContours(out, [hull], -1, 255, -1)
    # A whisker outwards for the blank margin outside the outermost panel.
    # Deliberately small: growing it further traded 1% of blank margin for
    # four times as much carpet, which is the wrong way round.
    g = max(3, int(min(sh, sw) * 0.012)) | 1
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))


def page_mask(img: np.ndarray) -> np.ndarray:
    """A full-size mask of the sheet: 255 on the page, 0 around it.

    Returns all-255 when it cannot find a page, so callers that treat the mask
    as "what to keep" leave such a photo alone.
    """
    h, w = img.shape[:2]
    k = min(1.0, WORK / float(max(h, w)))
    small = (cv2.resize(img, (max(8, int(w * k)), max(8, int(h * k))),
                        interpolation=cv2.INTER_AREA) if k < 1 else img.copy())
    sh, sw = small.shape[:2]
    keep_all = np.full((h, w), 255, np.uint8)

    # The model first, when it is installed: it traces the sheet's actual
    # outline, which no hand-built cue below can do.
    nn = _neural_keep(small)
    if nn is not None:
        return cv2.resize(nn, (w, h), interpolation=cv2.INTER_NEAREST)

    # First classical cue: the page is the busy region. Relative to this
    # photo's own detail, with a floor so that a picture of nothing much does
    # not have its noise promoted into "print".
    det = _detail(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    thr = max(6.0, float(np.percentile(det, 65)))
    out = _hull_of((det > thr).astype(np.uint8) * 255)

    # When the whole frame is "busy" — a carpet in focus is as busy as print —
    # the detail cue has failed, and the page is re-found as the largest
    # NEUTRAL region: paper is grey, a floor has colour.
    if out is None or int(cv2.countNonZero(out)) > FULL_FRAME * sh * sw:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].copy()
        # Ink votes neutral. HSV saturation is unstable over dark pixels —
        # a whisper of chroma noise on near-black reads as strongly coloured
        # — and it was carving the dense-dark artwork out of the page. Ink
        # sits on paper, so a dark pixel IS page evidence, and it says so
        # here by counting as perfectly neutral. (Lab chroma was tried
        # instead and is damped over dark pixels the opposite way: the
        # user's dark art scored 3.7–4.4 against carpet at 3.9–7.8 — no
        # separation at all.)
        sat[hsv[..., 2] < 100] = 0
        sat = cv2.blur(sat, (21, 21))
        _t, low = cv2.threshold(sat, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inside = sat[low > 0]
        outside = sat[low == 0]
        # Only trusted when the split found a real colour difference: on a
        # grey carpet Otsu just cuts noise in half, and that is no page.
        if (inside.size and outside.size
                and float(outside.mean()) - float(inside.mean()) >= 8.0):
            # A greyish patch of carpet passes the colour test too, and a
            # close would bridge the page to it and swallow the floor
            # between. So the specks are opened away and only page-sized
            # bodies are kept — all of them, for the spread whose two pages
            # meet only at the spine — and their JOINT convex hull is the
            # sheet. The hull matters here more than anywhere: where dark
            # art runs to the paper's edge the neutral region has a bite
            # taken out of it, and convexity is what guarantees the bite —
            # which is artwork — stays inside the cut. The price is a wedge
            # of carpet at a corner, and that is the right way round: a
            # tool that sometimes keeps a little floor is a tool, one that
            # sometimes eats the drawing is not. (For the same reason
            # nothing is subtracted INSIDE the hull however saturated it
            # is — a colour panel is still the page.)
            o = max(3, int(min(sh, sw) * 0.015)) | 1
            low = cv2.morphologyEx(low, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                             (o, o)))
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(low, 8)
            big = [i for i in range(1, n2)
                   if st2[i, cv2.CC_STAT_AREA] >= MIN_PAGE * sh * sw]
            if big:
                body = np.isin(lab2, big).astype(np.uint8)
                cnts, _hier = cv2.findContours(body, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    hull = cv2.convexHull(np.vstack(cnts))
                    alt = np.zeros((sh, sw), np.uint8)
                    cv2.drawContours(alt, [hull], -1, 255, -1)
                    g = max(3, int(min(sh, sw) * 0.012)) | 1
                    alt = cv2.dilate(alt, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (g, g)))
                    if int(cv2.countNonZero(alt)) <= FULL_FRAME * sh * sw:
                        out = alt

    if out is None or int(cv2.countNonZero(out)) > FULL_FRAME * sh * sw:
        return keep_all
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def engine() -> str:
    """Which cutter did the last cut: "ai" once the model has loaded and run,
    "basic" while it has not (not installed, or not called on yet)."""
    return "ai" if _NN not in (None, False) else "basic"


def cut_page(img: np.ndarray, background=(255, 255, 255), pad: int = 2):
    """Whiten everything that is not the page, and crop to the page.

    Returns (image, mask). The mask comes back too so a caller can offer the
    cut-out on transparency instead of on white.
    """
    m = page_mask(img)
    if int(cv2.countNonZero(m)) >= 0.995 * m.size:
        return img, m                      # nothing found, or nothing to cut
    out = img.copy()
    out[m == 0] = background
    x, y, w, h = cv2.boundingRect(m)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return img, m
    return out[y0:y1, x0:x1], m[y0:y1, x0:x1]

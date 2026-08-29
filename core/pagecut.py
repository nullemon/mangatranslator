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

    det = _detail(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    # Relative to this photo's own detail, with a floor so that a picture of
    # nothing much does not have its noise promoted into "print".
    thr = max(6.0, float(np.percentile(det, 65)))
    d = (det > thr).astype(np.uint8) * 255

    # Close the print into one body. The gaps being bridged are panel gutters
    # and balloons, which are page-sized, not pixel-sized.
    r = max(5, int(min(sh, sw) * 0.05)) | 1
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(d, 8)
    if n <= 1:
        return keep_all
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < MIN_PAGE * sh * sw:
        return keep_all

    cnts, _hier = cv2.findContours((lab == idx).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return keep_all
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros((sh, sw), np.uint8)
    cv2.drawContours(out, [hull], -1, 255, -1)

    # A whisker outwards for the blank margin outside the outermost panel.
    # Deliberately small: growing it further traded 1% of blank margin for
    # four times as much carpet, which is the wrong way round.
    g = max(3, int(min(sh, sw) * 0.012)) | 1
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


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

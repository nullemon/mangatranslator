#!/usr/bin/env python3
"""Rotate a pile of pages in one go.

    python3 tools/rotate.py 180 ~/chapter
    python3 tools/rotate.py 90  chapter.zip
    python3 tools/rotate.py 270 page1.png page2.png -o fixed/
    python3 tools/rotate.py 180 ~/chapter --in-place

Takes files, folders (searched right through) or a .zip, and writes to a
`rotated/` folder beside the input unless told otherwise. A zip in gives a zip
out.

Only quarter turns are offered, and that is deliberate: 90, 180 and 270 move
whole pixels to whole pixels, so the page that comes out is exactly the page
that went in, only turned. Any other angle has to invent pixels between the
old ones, which softens line art and leaves blank wedges in the corners — if
you ever need that, do it once on the page you are editing, not across a
chapter.

360 is accepted because it is the obvious way to ask for "leave it alone", and
it does exactly that: a copy, byte for byte.
"""
import argparse
import io
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

TURNS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    360: None,                      # a full turn is where you started
}


def turn(img, deg):
    """Rotate by a quarter turn. Exact — no resampling, nothing invented."""
    code = TURNS[deg]
    return img if code is None else cv2.rotate(img, code)


def encode(img, ext, quality):
    ext = ext if ext.lower() in IMG_EXT else ".png"
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext.lower() == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"could not encode {ext}")
    return buf.tobytes()


def gather(inputs):
    """Every image the arguments point at, in a sensible order."""
    files = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                for n in sorted(names):
                    if n.lower().endswith(IMG_EXT):
                        files.append(os.path.join(root, n))
        elif p.lower().endswith(".zip"):
            files.append(p)
        elif p.lower().endswith(IMG_EXT):
            files.append(p)
        else:
            print(f"  [--] skipped (not an image or zip): {p}")
    return files


def do_zip(src, deg, out_path, quality):
    n = 0
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            if name.endswith("/") or "__MACOSX" in name:
                continue
            if not name.lower().endswith(IMG_EXT):
                continue
            data = zin.read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [--] unreadable, copied as-is: {name}")
                zout.writestr(name, data)
                continue
            ext = os.path.splitext(name)[1]
            # A full turn is a copy: re-encoding would only lose quality for
            # no reason at all.
            out = data if deg == 360 else encode(turn(img, deg), ext, quality)
            zout.writestr(name, out)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Rotate a lot of pages at once (90, 180, 270 or 360).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Only quarter")[0].strip())
    ap.add_argument("degrees", type=int, choices=sorted(TURNS),
                    help="90 = clockwise, 270 = anticlockwise, 180 = upside "
                         "down, 360 = no change")
    ap.add_argument("inputs", nargs="+", help="images, folders or a .zip")
    ap.add_argument("-o", "--out", help="where to write (default: ./rotated "
                                        "beside the input)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the originals instead of writing copies")
    ap.add_argument("-q", "--quality", type=int, default=95,
                    help="JPEG/WebP quality when re-encoding (default 95)")
    args = ap.parse_args()

    files = gather(args.inputs)
    if not files:
        print("Nothing to do — no images or zips found.")
        return 1
    if args.in_place and args.out:
        print("Use --in-place or --out, not both.")
        return 2

    deg = args.degrees
    label = {90: "90° clockwise", 180: "180°",
             270: "90° anticlockwise", 360: "360° (no change)"}[deg]
    print(f"Rotating {len(files)} item(s) {label}\n")

    out_dir = args.out
    if not args.in_place and not out_dir:
        base = args.inputs[0]
        root = base if os.path.isdir(base) else os.path.dirname(os.path.abspath(base))
        out_dir = os.path.join(root, "rotated")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done = failed = 0
    for i, src in enumerate(files, 1):
        name = os.path.basename(src)
        try:
            if src.lower().endswith(".zip"):
                dest = (src if args.in_place
                        else os.path.join(out_dir, name))
                tmp = dest + ".part"
                n = do_zip(src, deg, tmp, args.quality)
                os.replace(tmp, dest)
                print(f"  [{i}/{len(files)}] {name}: {n} page(s) -> {dest}")
                done += n
                continue

            dest = src if args.in_place else os.path.join(out_dir, name)
            if deg == 360:
                # Nothing to change. Copy the bytes rather than decode and
                # re-encode them, which would only throw quality away.
                if dest != src:
                    shutil.copyfile(src, dest)
                done += 1
                continue
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [--] unreadable: {name}")
                failed += 1
                continue
            data = encode(turn(img, deg), os.path.splitext(name)[1], args.quality)
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)          # never leave a half-written page
            done += 1
            if len(files) <= 40 or i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {name}")
        except Exception as e:
            print(f"  [--] {name}: {e}")
            failed += 1

    where = "in place" if args.in_place else out_dir
    print(f"\nDone — {done} page(s) written to {where}" +
          (f", {failed} failed" if failed else ""))
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())

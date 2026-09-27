"""Translate pages through the RUNNING app, the way the browser does, and keep
everything needed to judge the result: the finished page, the original, the
detection report (/api/debug) and the task status.

    python3 app.py                       # in one terminal
    python3 tools/run_page.py raws/p-0073.jpg raws/p-0075.jpg --series "One Piece"
    python3 tools/run_page.py raws/ --series "One Piece" --set max_quality=true

The translator key is read from the environment or .env (GEMINI_API_KEY, or
ANTHROPIC_API_KEY with --provider claude), so it never appears on the
command line or in shell history. Results go to output/runs/<time>/.
Any form field the browser sends can be set with --set name=value
(see the /api/translate parameters in app.py)."""
import argparse
import json
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_env(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


def pages(args):
    out = []
    for p in args:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, n) for n in os.listdir(p)
                          if n.lower().endswith(IMAGE_EXT))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", help="page images or folders of them")
    ap.add_argument("--url", default=os.environ.get("MT_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--provider", default="gemini", choices=["gemini", "claude", "local"])
    ap.add_argument("--model", default=os.environ.get("MT_MODEL", ""))
    ap.add_argument("--series", default="", help='manga title, e.g. "One Piece"')
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="extra /api/translate form field (repeatable)")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per page")
    a = ap.parse_args()

    load_env(os.path.join(ROOT, ".env"))
    key_var = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(a.provider)
    key = os.environ.get(key_var, "") if key_var else ""
    if key_var and not key:
        sys.exit(f"Set {key_var} in .env (or the environment) first.")

    form = {"provider": a.provider, "api_key": key, "target_lang": "English"}
    if a.model:
        form["model"] = a.model
    if a.series:
        # the same line the browser's Manga title box adds
        form["style_prompt"] = (
            f'SERIES: this page is from "{a.series}". Use that manga\'s canonical '
            "character names, place names, techniques and tone exactly as known "
            "from the series — never invent or re-romanize them.")
    for kv in a.set:
        k, _, v = kv.partition("=")
        form[k.strip()] = v

    out_dir = a.out or os.path.join(ROOT, "output", "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    todo = pages(a.pages)
    if not todo:
        sys.exit("no page images given")
    summary = []
    # the app is local: never send its requests through a system proxy
    local = any(h in a.url for h in ("127.0.0.1", "localhost", "0.0.0.0"))
    with httpx.Client(base_url=a.url, timeout=120, trust_env=not local) as c:
        try:
            health = c.get("/api/health").json()
            print(f"server build {health.get('server_commit', '?')}")
        except Exception as e:
            sys.exit(f"The app isn't answering at {a.url} ({e}). Start it: python3 app.py")
        for path in todo:
            name = os.path.splitext(os.path.basename(path))[0]
            t0 = time.time()
            with open(path, "rb") as f:
                r = c.post("/api/translate", data=form,
                           files={"file": (os.path.basename(path), f, "image/jpeg"
                                           if path.lower().endswith((".jpg", ".jpeg"))
                                           else "image/png")})
            if r.status_code != 200:
                print(f"{name}: translate refused ({r.status_code}) {r.text[:200]}")
                summary.append({"page": name, "status": f"http {r.status_code}"})
                continue
            tid = r.json()["task_id"]
            st = {}
            while time.time() - t0 < a.timeout:
                st = c.get(f"/api/status/{tid}").json()
                if st.get("status") in ("done", "error"):
                    break
                time.sleep(2)
            secs = time.time() - t0
            with open(os.path.join(out_dir, f"{name}.status.json"), "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=1)
            dbg = c.get(f"/api/debug/{tid}").text
            with open(os.path.join(out_dir, f"{name}.debug.txt"), "w", encoding="utf-8") as f:
                f.write(dbg)
            if st.get("status") == "done":
                for route, suffix in ((f"/api/result/{tid}", "out"),
                                      (f"/api/original/{tid}", "raw")):
                    rr = c.get(route)
                    ext = ".jpg" if "jpeg" in rr.headers.get("content-type", "") else ".png"
                    with open(os.path.join(out_dir, f"{name}.{suffix}{ext}"), "wb") as f:
                        f.write(rr.content)
            print(f"{name}: {st.get('status', 'timeout')} in {secs:.0f}s  "
                  f"({dbg.count(chr(10)) - 2} lines in the debug report)")
            summary.append({"page": name, "task_id": tid, "status": st.get("status"),
                            "seconds": round(secs), "error": st.get("error")})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()

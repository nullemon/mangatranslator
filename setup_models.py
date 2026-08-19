#!/usr/bin/env python3
"""Cross-platform AI model setup for MangaTranslator.

Installs the local models that make detection, cleanup and the
text-verification logic accurate:

  - speech-balloon segmentation (precise bubble masks)
  - manga-ocr               (reads each bubble's own text -> translations
                             can never land in the wrong bubble, and fake
                             regions like faces/shirts are refused)
  - comic-text-detector     (text-stroke pixel masks -> surgical erasure,
                             second verification signal)
  - LaMa inpainting         (clean text removal over art)
  - CRAFT                   (free-text detection)
  - spandrel                (upscaler runtime)

Everything here runs WITHOUT a GPU — on CPU it is slower but the same
quality and the same verification logic. With an NVIDIA GPU, pass --gpu
for the CUDA builds.

Usage:
  python setup_models.py            # install for this machine (CPU/any OS)
  python setup_models.py --gpu      # NVIDIA CUDA builds (Windows/Linux)
  python setup_models.py --check    # only report what's present / missing
  python setup_models.py --dry-run  # print the commands without running
  python setup_models.py --offline-translate
                                    # OFFLINE translation packs (~300MB each)
                                    # so pages translate on this PC with no
                                    # API key and no internet. Japanese and
                                    # Korean by default.
  python setup_models.py --offline-translate --langs all
                                    # every language pack
  python setup_models.py --offline-translate --langs japanese,chinese
                                    # just the ones you read
"""
import importlib.util
import os
import platform
import subprocess
import sys

CHECK = "--check" in sys.argv
DRY = "--dry-run" in sys.argv
GPU = "--gpu" in sys.argv
OFFLINE_MT = "--offline-translate" in sys.argv


def _arg_list(flag):
    """--langs all   /   --langs japanese,korean   /   --langs=japanese"""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return [s.strip().lower() for s in sys.argv[i + 1].split(",") if s.strip()]
        if a.startswith(flag + "="):
            return [s.strip().lower() for s in a.split("=", 1)[1].split(",") if s.strip()]
    return []


OFFLINE_LANGS = _arg_list("--langs")
IS_MAC = platform.system() == "Darwin"

MODULES = {
    "torch": "PyTorch (runs the AI models)",
    "ultralytics": "YOLO runtime (balloon segmentation)",
    "huggingface_hub": "model downloads",
    "manga_ocr": "manga-ocr (per-bubble reading)",
    "simple_lama_inpainting": "LaMa (clean text erasure)",
    "onnxruntime": "ONNX runtime (text-stroke model)",
    "craft_text_detector": "CRAFT (free-text detection)",
    "spandrel": "upscaler runtime",
    "scipy": "CRAFT dependency",
}


def have(mod):
    return importlib.util.find_spec(mod) is not None


def pip(args):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print("  $ " + " ".join(cmd[3:]))
    if DRY:
        return True
    return subprocess.call(cmd) == 0


def main():
    print("MangaTranslator model setup")
    print(f"platform: {platform.system()} {platform.machine()}, "
          f"python {sys.version.split()[0]}\n")

    missing = [(m, d) for m, d in MODULES.items() if not have(m)]
    for m, d in MODULES.items():
        print(f"  [{'ok' if have(m) else '--'}] {m:<24} {d}")
    if CHECK:
        print(f"\n{len(missing)} missing." if missing else "\nAll present.")
        return
    if not missing:
        print("\nEverything is already installed.")
    print()

    steps_ok = True

    # 1) torch — the base runtime for most models.
    if not have("torch"):
        print("[1/4] PyTorch...")
        if GPU and not IS_MAC:
            steps_ok &= pip(["torch", "torchvision",
                             "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            # Default wheels: CPU on Windows, CPU+MPS on macOS, CUDA-capable
            # on Linux. All fine — the app auto-detects at runtime.
            steps_ok &= pip(["torch", "torchvision"])
    else:
        print("[1/4] PyTorch already installed.")

    # 2) the model packages — only the ones actually missing, so a complete
    #    install is a clean no-op (and PEP-668 distros aren't poked at all).
    pkg_map = {"ultralytics": "ultralytics", "huggingface_hub": "huggingface_hub",
               "manga_ocr": "manga-ocr",
               "simple_lama_inpainting": "simple-lama-inpainting",
               "scipy": "scipy", "gdown": "gdown", "spandrel": "spandrel"}
    need = [p for m, p in pkg_map.items() if not have(m)]
    if need:
        print("[2/4] Model packages...")
        steps_ok &= pip(need)
    else:
        print("[2/4] Model packages already installed.")

    # 3) ONNX runtime for the text-stroke model. Never clobber an existing
    #    -gpu install; macOS has no -gpu build at all.
    if not have("onnxruntime"):
        print("[3/4] ONNX runtime...")
        if GPU and not IS_MAC:
            steps_ok &= pip(["onnxruntime-gpu"])
        else:
            steps_ok &= pip(["onnxruntime"])
    else:
        print("[3/4] ONNX runtime already installed.")

    # 4) CRAFT — installed without deps on purpose: its setup pins an ancient
    #    opencv that no longer builds; the real runtime deps are covered above.
    if not have("craft_text_detector"):
        print("[4/4] CRAFT free-text detector...")
        steps_ok &= pip(["craft-text-detector", "--no-deps"])
    else:
        print("[4/4] CRAFT already installed.")

    if DRY:
        print("\n(dry run — nothing was installed)")
        return

    # Pre-download the weights so the first page isn't a long wait.
    print("\nPre-downloading model weights (best effort)...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core.text_seg import _weights_path, _download_weights
        p = _weights_path()
        if os.path.exists(p) or _download_weights(p):
            print("  [ok] text-stroke model")
        else:
            print("  [--] text-stroke model (will retry on first run)")
    except Exception as e:
        print(f"  [--] text-stroke model: {e}")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="kitsumed/yolov8m_seg-speech-bubble",
                        filename="model.pt")
        print("  [ok] balloon segmentation model")
    except Exception as e:
        print(f"  [--] balloon model: {e}")
    print("  (manga-ocr and LaMa download themselves on first use)")

    if OFFLINE_MT:
        print("\nOffline translation packs (no API key, no internet)...")
        pip(["sentencepiece", "sacremoses"])
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from core.local_mt import (get, model_id_for, DEFAULT_MODELS,
                                       COMMON_LANGS, installed_langs)
            already = set(installed_langs())   # NB: `have` is a module fn
            if OFFLINE_LANGS == ["all"]:
                langs = list(DEFAULT_MODELS)
            elif OFFLINE_LANGS:
                langs = OFFLINE_LANGS
            else:
                langs = list(COMMON_LANGS)
            unknown = [l for l in langs if l not in DEFAULT_MODELS]
            for u in unknown:
                print(f"  [--] no pack for {u!r}. Available: "
                      f"{', '.join(sorted(DEFAULT_MODELS))}")
            langs = [l for l in langs if l in DEFAULT_MODELS]
            todo = [l for l in langs if l not in already]
            for l in langs:
                if l in already:
                    print(f"  [ok] {l} already downloaded")
            if todo:
                print(f"  downloading {len(todo)} pack(s), "
                      f"about {0.3 * len(todo):.1f}GB total")
            for lang in todo:
                print(f"  -> {lang}: {model_id_for(lang)}")
                if get(lang) is None:
                    print(f"  [--] {lang} failed — retry, or use an API engine")
                else:
                    print(f"  [ok] {lang} ready")
        except Exception as e:
            print(f"  [--] offline translation setup failed: {e}")
    else:
        print("\nTip: --offline-translate downloads the on-device translation")
        print("     packs (no API key, no internet, much faster). By default")
        print("     that is Japanese and Korean; add --langs all for every")
        print("     language, or --langs japanese,chinese to pick your own.")

    print("\nDone." if steps_ok else
          "\nFinished with some failures — re-run, or check your internet "
          "connection. The app still works; failed pieces fall back to CV.")
    print("Start the app with: python app.py  — the startup banner shows "
          "which stages are active. `python check_setup.py` gives a full "
          "report.")


if __name__ == "__main__":
    main()

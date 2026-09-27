"""The page runner reads each page's cost from the app's note, so a paid run
can stop at its cap. A free page (no note) is $0; a note the app couldn't
price (tokens only) is None, which the runner treats as "assume the worst".

Run:  python tests/test_run_page_cost.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from run_page import page_cost  # noqa: E402


def main():
    assert page_cost({"cost_note": "≈ $0.012 · 3 AI calls"}) == 0.012
    assert page_cost({"cost_note": "≈ $0.000 · 1 AI call"}) == 0.0
    assert page_cost({"cost_note": ""}) == 0.0
    assert page_cost({}) == 0.0
    assert page_cost({"cost_note": "2,400 tokens · 2 AI calls"}) is None
    print("cost notes read OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

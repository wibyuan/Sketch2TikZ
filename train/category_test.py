"""
Per-category quick test: 1 sample, universal vs category.
Usage:
    python -m train.category_test chart_plot
    python -m train.category_test math_formula
    python -m train.category_test math_geometry
    python -m train.category_test pure_drawing
"""
import os, sys, glob, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(override=True)

from train.pipeline import generate as pipeline_generate
from train.pipeline import _internal_critic


def safe_print(*a, **kw):
    try:
        print(*a, **kw)
    except UnicodeEncodeError:
        print(*(str(x).encode("ascii", "replace").decode("ascii") for x in a), **kw)


def run_one(cat: str, out_dir: str):
    cat_dir = os.path.join("content_category_v2", cat)
    imgs = sorted(glob.glob(os.path.join(cat_dir, "sample_*.png")))[:1]
    if not imgs:
        safe_print(f"No images found for {cat}")
        return None

    png = imgs[0]
    safe_print(f"\n=== {cat} ===  {os.path.basename(png)}")

    # Universal
    safe_print("\n[UNIVERSAL prompt]")
    t0 = time.time()
    r1 = pipeline_generate(png, 0, output_dir=os.path.join(out_dir, cat, "universal"), category=None)
    t1 = time.time()
    c1 = _internal_critic(png, r1.gen_pdf_path, os.path.join(out_dir, cat, "universal")) if r1.compile_ok else None
    score1 = c1["score"] if c1 else 0.0
    safe_print(f"  Compile: {'OK' if r1.compile_ok else 'FAIL'} | Score: {score1:.1f} | {c1.get('diagnosis','')[:80] if c1 else ''}")

    # Category
    safe_print("\n[CATEGORY prompt]")
    t2 = time.time()
    r2 = pipeline_generate(png, 0, output_dir=os.path.join(out_dir, cat, "category"), category=cat)
    t3 = time.time()
    c2 = _internal_critic(png, r2.gen_pdf_path, os.path.join(out_dir, cat, "category")) if r2.compile_ok else None
    score2 = c2["score"] if c2 else 0.0
    safe_print(f"  Compile: {'OK' if r2.compile_ok else 'FAIL'} | Score: {score2:.1f} | {c2.get('diagnosis','')[:80] if c2 else ''}")

    delta = score2 - score1
    safe_print(f"\n  DELTA: {delta:+.1f}  (universal {score1:.1f} -> category {score2:.1f})")

    return {
        "cat": cat,
        "png": os.path.basename(png),
        "universal": {"compile_ok": r1.compile_ok, "score": score1, "diagnosis": c1.get("diagnosis","") if c1 else ""},
        "category": {"compile_ok": r2.compile_ok, "score": score2, "diagnosis": c2.get("diagnosis","") if c2 else ""},
        "delta": delta,
    }


if __name__ == "__main__":
    cat = sys.argv[1] if len(sys.argv) > 1 else "chart_plot"
    out_dir = os.path.join("cat_bench", "quick")
    result = run_one(cat, out_dir)
    if result:
        json_path = os.path.join(out_dir, f"{cat}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        safe_print(f"\nSaved: {json_path}")

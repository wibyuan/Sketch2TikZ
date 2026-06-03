"""
A/B benchmark: universal prompt vs. category-specialised prompt
on content_category_v2 samples.

Usage:
    python -m train.category_benchmark --samples-per-cat 3
    python -m train.category_benchmark --samples-per-cat 5 --output-dir cat_bench
"""
import os, sys, glob, time, json, argparse
from datetime import datetime

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


CATEGORIES = ["chart_plot", "math_formula", "math_geometry", "pure_drawing"]
DATA_ROOT = "content_category_v2"


def run_category(cat: str, num_samples: int, use_category_prompt: bool, out_dir: str):
    """Run pipeline on N samples of a category. Returns list of result dicts."""
    cat_dir = os.path.join(DATA_ROOT, cat)
    imgs = sorted(glob.glob(os.path.join(cat_dir, "sample_*.png")))[:num_samples]

    prompt_type = "CATEGORY" if use_category_prompt else "UNIVERSAL"
    safe_print(f"\n>>> [{cat}] {prompt_type} prompt  (n={len(imgs)})")

    results = []
    for i, png in enumerate(imgs):
        t0 = time.time()
        r = pipeline_generate(
            png, i,
            output_dir=os.path.join(out_dir, cat, prompt_type.lower()),
            category=cat if use_category_prompt else None,
        )
        elapsed = round(time.time() - t0, 1)

        critic = None
        if r.compile_ok and r.gen_pdf_path and os.path.exists(r.gen_pdf_path):
            critic = _internal_critic(png, r.gen_pdf_path, os.path.join(out_dir, cat, prompt_type.lower()))

        record = {
            "basename": os.path.basename(png),
            "compile_ok": r.compile_ok,
            "compile_attempts": r.compile_attempts,
            "critic_score": critic["score"] if critic else 0.0,
            "critic_pass": critic["is_pass"] if critic else False,
            "critic_diagnosis": critic.get("diagnosis", "") if critic else "",
            "vision_time": r.vision_time,
            "codegen_time": r.codegen_time,
            "elapsed_total": elapsed,
        }
        results.append(record)

        score = record["critic_score"]
        diag = record["critic_diagnosis"]
        safe_print(f"  [{i+1}/{len(imgs)}] {record['basename']} | "
                   f"Compile: {'OK' if r.compile_ok else 'FAIL'} ({r.compile_attempts} att) | "
                   f"Score: {score:.1f} | {diag[:100]}")

    return results


def aggregate(results: list) -> dict:
    n = len(results)
    compile_ok = sum(1 for r in results if r["compile_ok"])
    judged = [r for r in results if r["compile_ok"]]
    scores = [r["critic_score"] for r in judged]
    passes = sum(1 for r in judged if r["critic_pass"])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_attempts = sum(r["compile_attempts"] for r in results) / n if n else 0.0
    avg_vision = sum(r["vision_time"] for r in results) / n if n else 0.0
    avg_codegen = sum(r["codegen_time"] for r in results) / n if n else 0.0
    return {
        "n": n,
        "compile_ok": compile_ok,
        "compile_rate": 100 * compile_ok / n if n else 0.0,
        "passes": passes,
        "pass_rate": 100 * passes / max(1, len(judged)),
        "avg_score": avg_score,
        "avg_attempts": avg_attempts,
        "avg_vision_time": avg_vision,
        "avg_codegen_time": avg_codegen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-cat", type=int, default=3,
                        help="Samples per category (max 10)")
    parser.add_argument("--output-dir", default="",
                        help="Output directory (default: cat_bench/{timestamp})")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join("cat_bench", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    safe_print("=" * 70)
    safe_print("CATEGORY PROMPT A/B BENCHMARK")
    safe_print("=" * 70)
    safe_print(f"Samples per category: {args.samples_per_cat}")
    safe_print(f"Categories: {', '.join(CATEGORIES)}")
    safe_print(f"Output: {out_dir}")
    safe_print()

    report = {}

    for cat in CATEGORIES:
        safe_print(f"\n{'='*70}")
        safe_print(f"CATEGORY: {cat}")
        safe_print("=" * 70)

        # Baseline (universal)
        baseline_results = run_category(cat, args.samples_per_cat, use_category_prompt=False, out_dir=out_dir)
        baseline_agg = aggregate(baseline_results)

        # Category-specialised
        cat_results = run_category(cat, args.samples_per_cat, use_category_prompt=True, out_dir=out_dir)
        cat_agg = aggregate(cat_results)

        report[cat] = {
            "baseline": {"results": baseline_results, "aggregate": baseline_agg},
            "category": {"results": cat_results, "aggregate": cat_agg},
        }

        # Per-category summary
        safe_print(f"\n  [{cat}] SUMMARY")
        safe_print(f"    Universal  -> compile: {baseline_agg['compile_ok']}/{baseline_agg['n']} "
                   f"score: {baseline_agg['avg_score']:.2f} pass: {baseline_agg['pass_rate']:.0f}%")
        safe_print(f"    Category   -> compile: {cat_agg['compile_ok']}/{cat_agg['n']} "
                   f"score: {cat_agg['avg_score']:.2f} pass: {cat_agg['pass_rate']:.0f}%")
        delta = cat_agg['avg_score'] - baseline_agg['avg_score']
        safe_print(f"    Delta score: {delta:+.2f}")

    # ── Global aggregate ──
    safe_print(f"\n{'='*70}")
    safe_print("GLOBAL SUMMARY")
    safe_print("=" * 70)

    all_baseline = []
    all_category = []
    for cat in CATEGORIES:
        all_baseline.extend(report[cat]["baseline"]["results"])
        all_category.extend(report[cat]["category"]["results"])

    global_base = aggregate(all_baseline)
    global_cat = aggregate(all_category)

    safe_print(f"{'Category':<18} {'Uni-Score':>10} {'Cat-Score':>10} {'Delta':>8} {'Uni-Pass':>10} {'Cat-Pass':>10}")
    safe_print("-" * 70)
    for cat in CATEGORIES:
        b = report[cat]["baseline"]["aggregate"]
        c = report[cat]["category"]["aggregate"]
        safe_print(f"{cat:<18} {b['avg_score']:>10.2f} {c['avg_score']:>10.2f} "
                   f"{c['avg_score']-b['avg_score']:>+8.2f} {b['pass_rate']:>9.0f}% {c['pass_rate']:>9.0f}%")
    safe_print("-" * 70)
    safe_print(f"{'TOTAL':<18} {global_base['avg_score']:>10.2f} {global_cat['avg_score']:>10.2f} "
               f"{global_cat['avg_score']-global_base['avg_score']:>+8.2f} "
               f"{global_base['pass_rate']:>9.0f}% {global_cat['pass_rate']:>9.0f}%")

    # Save JSON
    json_path = os.path.join(out_dir, "benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    safe_print(f"\nFull results saved: {json_path}")


if __name__ == "__main__":
    main()

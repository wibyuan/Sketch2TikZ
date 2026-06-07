"""
Iterative optimisation loop on TRAIN data (ground-truth TikZ available).

Usage:
    python -m train.dev_loop --difficulty easy --samples 5 --review
    python -m train.dev_loop --difficulty easy --samples 5 --review --threshold 2.0

Flow:
    1. Run pipeline on N train samples (PNG + GT .tikz)
    2. Compile + internal critic (reference = original PNG)
    3. For samples with critic_score < threshold: AI academic review
       (vision model sees: sketch + generated render + GT code + gen code + diagnosis)
    4. Aggregate reviews into a structured report with deduplicated prompt fixes
    5. Save everything to iter_runs/{timestamp}/ for traceability
"""
import os, sys, glob, time, json, textwrap, argparse, shutil
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(override=True)

from train.pipeline import generate as pipeline_generate
from train.pipeline import _internal_critic
from train.askai import review as ai_review
from train.prompts import load_prompts


def safe_print(*a, **kw):
    try:
        print(*a, **kw)
    except UnicodeEncodeError:
        print(*(str(x).encode("ascii", "replace").decode("ascii") for x in a), **kw)


def _load_gt_code(png_path: str) -> str:
    tikz_path = png_path.replace(".png", ".tikz")
    if not os.path.exists(tikz_path):
        return ""
    with open(tikz_path, "r", encoding="utf-8") as f:
        return f.read()


def _shorten_code(code: str, limit: int = 2000) -> str:
    if len(code) <= limit:
        return code
    return code[:limit] + "\n... (truncated)\n"


def run_evaluation(
    difficulty: str,
    num_samples: int,
    threshold: float,
    do_review: bool,
    out_dir: str,
):
    train_dir = os.path.join(os.path.dirname(__file__), "data", difficulty)
    imgs = sorted(glob.glob(os.path.join(train_dir, "????.png")))[:num_samples]

    safe_print(f"=== Iterative Dev Loop ===")
    safe_print(f"Branch:  iter-opt")
    safe_print(f"Data:    train/{difficulty}  (n={len(imgs)})")
    safe_print(f"Review:  {'ON' if do_review else 'OFF'}  (threshold={threshold})")
    safe_print(f"Output:  {out_dir}")
    safe_print()

    os.makedirs(out_dir, exist_ok=True)

    # Save current prompts for reproducibility
    prompts = load_prompts(difficulty)
    with open(os.path.join(out_dir, "prompts_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump({
            "vision_prompt": prompts["vision_prompt"],
            "code_system": prompts["code_system"],
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    samples = []
    for i, png in enumerate(imgs):
        basename = os.path.basename(png)
        safe_print(f"[{i+1}/{len(imgs)}] {basename}")
        t0 = time.time()

        # --- Generate ---
        r = pipeline_generate(png, i, output_dir=out_dir)
        elapsed = round(time.time() - t0, 1)

        # --- Load GT ---
        gt_code = _load_gt_code(png)

        # --- Critic (reference = original sketch) ---
        critic = None
        if r.compile_ok and r.gen_pdf_path and os.path.exists(r.gen_pdf_path):
            critic = _internal_critic(png, r.gen_pdf_path, out_dir)

        # --- Record ---
        record = {
            "index": i,
            "basename": basename,
            "gt_code": gt_code,
            "gen_code": _shorten_code(open(r.gen_pdf_path.replace(".pdf", ".tex"), "r", encoding="utf-8").read()) if r.compile_ok else "",
            "compile_ok": r.compile_ok,
            "compile_attempts": r.compile_attempts,
            "vision_time": r.vision_time,
            "codegen_time": r.codegen_time,
            "critic_score": critic["score"] if critic else 0.0,
            "critic_pass": critic["is_pass"] if critic else False,
            "critic_diagnosis": critic.get("diagnosis", "") if critic else "",
            "elapsed_total": elapsed,
            "ai_review": None,
        }

        score = record["critic_score"]
        diag = record["critic_diagnosis"]
        safe_print(f"  Compile: {'OK' if r.compile_ok else 'FAIL'} ({r.compile_attempts} att) | "
                   f"Critic: {score:.1f}/5.0 | {diag[:120]}")

        # --- AI Review for low-score samples ---
        if do_review and critic and score < threshold:
            safe_print(f"  → Triggering AI review (score {score} < {threshold})")
            try:
                review = ai_review(
                    original_png=png,
                    gen_pdf=r.gen_pdf_path,
                    gt_code=gt_code,
                    gen_code=record["gen_code"],
                    critic_score=score,
                    critic_diagnosis=diag,
                    out_dir=out_dir,
                )
                record["ai_review"] = review
                if "error" not in review:
                    safe_print(f"    Root cause: {review.get('root_cause', 'N/A')}")
                    safe_print(f"    Severity:   {review.get('severity', 'N/A')} | Confidence: {review.get('confidence', 'N/A')}")
                else:
                    safe_print(f"    AI review error: {review.get('error', 'unknown')}")
            except Exception as e:
                safe_print(f"    AI review exception: {e}")
                record["ai_review"] = {"error": str(e)}

        samples.append(record)
        safe_print()

    # --- Aggregate report ---
    safe_print("=" * 60)
    safe_print("AGGREGATE REPORT")
    safe_print("=" * 60)

    compile_ok = sum(1 for s in samples if s["compile_ok"])
    judged = [s for s in samples if s["compile_ok"]]
    scores = [s["critic_score"] for s in judged]
    passes = sum(1 for s in judged if s["critic_pass"])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_attempts = sum(s["compile_attempts"] for s in samples) / len(samples) if samples else 0.0

    safe_print(f"  Samples:            {len(samples)}")
    safe_print(f"  Compile rate:       {compile_ok}/{len(samples)} ({100*compile_ok/len(samples):.1f}%)")
    safe_print(f"  Avg compile att:    {avg_attempts:.1f}")
    safe_print(f"  Pass rate (≥3.0):   {passes}/{max(1,len(judged))} ({100*passes/max(1,len(judged)):.1f}%)")
    safe_print(f"  Avg fidelity:       {avg_score:.2f} / 5.0")
    safe_print()

    # --- AI review aggregation ---
    reviews = [s["ai_review"] for s in samples if s["ai_review"] and "error" not in s["ai_review"]]
    if reviews:
        safe_print(f"AI reviews collected: {len(reviews)}")
        safe_print()

        by_severity = {"critical": [], "major": [], "minor": []}
        for rv in reviews:
            sev = rv.get("severity", "major").lower()
            by_severity.setdefault(sev, []).append(rv)

        report_lines = []
        report_lines.append("# Iteration Report\n")
        report_lines.append(f"**Date:** {datetime.now().isoformat()}\n")
        report_lines.append(f"**Branch:** iter-opt\n")
        report_lines.append(f"**Data:** train/{difficulty}  n={len(samples)}\n")
        report_lines.append(f"**Metrics:** compile={compile_ok}/{len(samples)} pass={passes}/{len(judged)} avg_score={avg_score:.2f}\n")
        report_lines.append("\n---\n")

        # Deduplicated fixes
        vision_fixes = []
        code_fixes = []
        example_fixes = []

        for sev in ["critical", "major", "minor"]:
            items = by_severity.get(sev, [])
            if not items:
                continue
            report_lines.append(f"\n## {sev.upper()} issues ({len(items)})\n")
            for rv in items:
                report_lines.append(f"\n### {rv.get('root_cause', 'Unknown')}\n")
                report_lines.append(f"- **Confidence:** {rv.get('confidence', 'N/A')}\n")
                report_lines.append(f"- **Rationale:** {rv.get('rationale', 'N/A')}\n")
                if rv.get("vision_prompt_fix"):
                    report_lines.append(f"- **Vision fix:** `{rv['vision_prompt_fix']}`\n")
                    vision_fixes.append(rv["vision_prompt_fix"])
                if rv.get("code_system_fix"):
                    report_lines.append(f"- **Code fix:** `{rv['code_system_fix']}`\n")
                    code_fixes.append(rv["code_system_fix"])
                if rv.get("example_fix"):
                    report_lines.append(f"- **Example fix:** `{rv['example_fix']}`\n")
                    example_fixes.append(rv["example_fix"])

        report_lines.append("\n---\n")
        report_lines.append("## Recommended Prompt Changes (deduplicated)\n")

        report_lines.append("\n### Add to VISION_PROMPT:\n")
        seen = set()
        for fix in vision_fixes:
            if fix not in seen:
                seen.add(fix)
                report_lines.append(f"- {fix}\n")
        if not vision_fixes:
            report_lines.append("_None suggested_\n")

        report_lines.append("\n### Add to CODE_SYSTEM:\n")
        seen = set()
        for fix in code_fixes:
            if fix not in seen:
                seen.add(fix)
                report_lines.append(f"- {fix}\n")
        if not code_fixes:
            report_lines.append("_None suggested_\n")

        report_lines.append("\n### Add to EXAMPLES:\n")
        seen = set()
        for fix in example_fixes:
            if fix not in seen:
                seen.add(fix)
                report_lines.append(f"- {fix}\n")
        if not example_fixes:
            report_lines.append("_None suggested_\n")

        report_md = "".join(report_lines)
        report_path = os.path.join(out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        safe_print(report_md)
        safe_print(f"\nReport saved: {report_path}")
    else:
        safe_print("No AI reviews collected (all scores above threshold or review disabled).")

    # --- Save raw JSON ---
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    safe_print(f"Raw JSON saved: {json_path}")
    safe_print()
    safe_print("Done. Edit train/prompts.py based on the report, then re-run to measure improvement.")


def main():
    parser = argparse.ArgumentParser(description="Iterative dev loop on train data")
    parser.add_argument("--difficulty", choices=["easy", "medium", "difficult"], default="easy")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Critic score below this triggers AI review")
    parser.add_argument("--review", action="store_true", default=True,
                        help="Enable AI academic review for low-score samples")
    parser.add_argument("--no-review", dest="review", action="store_false",
                        help="Skip AI review (faster, metrics only)")
    parser.add_argument("--output-dir", default="",
                        help="Override output directory (default: iter_runs/{timestamp})")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("iter_runs", f"{args.difficulty}_{ts}")

    run_evaluation(
        difficulty=args.difficulty,
        num_samples=args.samples,
        threshold=args.threshold,
        do_review=args.review,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

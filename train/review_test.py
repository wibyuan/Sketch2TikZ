"""Review results via AI.

Default mode (test data, no GT): visual pattern analysis only.
Use --train to run on train data with ground-truth code — this enables
both visual AND structural code-level review for diagnostic purposes.
"""
import os, sys, glob, time, json, base64, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(override=True)

from train.llm_caller import _create, VISION_MODELS, VISION_PLATFORMS
from train.pipeline import generate as pipeline_generate
from train.code_reviewer import review_code

REVIEW_PROMPT = """You are a TikZ quality inspector. Given:
1. A REFERENCE image
2. GENERATED code and its compile status
3. INTERNAL CRITIC score + diagnosis

Identify PATTERNS of failure across samples. Output JSON:
{
  "patterns": ["pattern1", "pattern2"],
  "vision_prompt_fix": "<concrete sentence>",
  "code_system_fix": "<concrete rule>",
  "severity": "low|medium|high"
}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run on train data (enables GT code review)")
    parser.add_argument("--difficulty", default="easy", choices=["easy","medium","difficult"])
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    if args.train:
        data_dir = os.path.join(os.path.dirname(__file__), "data", args.difficulty)
        print(f"=== Train Review ({args.difficulty}, {args.num_samples} samples, GT code enabled) ===")
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "test", "data", args.difficulty)
        print(f"=== Test Review ({args.difficulty}, {args.num_samples} samples, no GT) ===")

    imgs = sorted(glob.glob(os.path.join(data_dir, "????.png")))[:args.num_samples]

    results = []
    for i, png in enumerate(imgs):
        print(f"[{i+1}/{len(imgs)}] {os.path.basename(png)}")
        t0 = time.time()
        r = pipeline_generate(png, i, output_dir="output")
        elapsed = time.time() - t0
        status = "OK" if r.compile_ok else "FAIL"
        score = r.critic_score
        diagnosis = r.diagnosis
        print(f"  Compile: {status} ({r.compile_attempts} att) | Critic: {score:.1f} | {elapsed:.1f}s")
        if r.compile_ok and score < 4.0:
            print(f"  Diagnosis: {diagnosis[:150]}")

        # --- Train-mode: structural code review with GT ---
        code_review = None
        if args.train:
            gt_path = png.replace(".png", ".tikz")
            if os.path.exists(gt_path) and r.compile_ok:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_code = f.read()
                gen_tex = os.path.join("output", f"gen_{i:04d}.tex")
                gen_code = ""
                if os.path.exists(gen_tex):
                    with open(gen_tex, "r", encoding="utf-8") as f:
                        gen_code = f.read()
                if gt_code and gen_code:
                    print(f"  → Running structural code review...")
                    code_review = review_code(gt_code, gen_code)
                    if "error" not in code_review:
                        cs = code_review.get("code_score", 0)
                        sev = code_review.get("severity", "N/A")
                        print(f"    Code score: {cs}/5.0 | Severity: {sev}")
                        if code_review.get("structural_hacks"):
                            print(f"    Hacks: {code_review['structural_hacks']}")
                    else:
                        print(f"    Code review error: {code_review.get('error', 'unknown')}")

        results.append({"png": png, "r": r, "score": score, "diagnosis": diagnosis, "code_review": code_review})
        print()

    # AI Review (visual patterns)
    print("=== AI Pattern Analysis ===")
    bad = [s for s in results if s["r"].compile_ok and s["score"] < 4.0]
    if not bad:
        print("All samples >= 4.0. No patterns to fix.")
    else:
        print(f"Reviewing {len(bad)} low-score sample(s)")
        inputs = []
        for s in bad:
            base = os.path.basename(s["png"])
            tex = os.path.join("output", f"gen_{s['r'].index:04d}.tex")
            code = ""
            if os.path.exists(tex):
                with open(tex, encoding="utf-8") as f:
                    code = f.read()[:2000]
            inputs.append(f"Sample {base}: Critic={s['score']:.1f}. {s['diagnosis']}\nCode:\n{code[:500]}")

        review_text = "PATTERNS ACROSS FAILURES:\n" + "\n---\n".join(inputs)
        platform = VISION_PLATFORMS[0]
        model = VISION_MODELS[platform]

        # Include worst-scoring image for visual context
        worst = min(bad, key=lambda x: x["score"])
        from train.pipeline import _encode_img
        b64 = _encode_img(worst["png"])

        try:
            raw = _create(platform, model, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64}},
                {"type": "text", "text": review_text},
                {"type": "text", "text": REVIEW_PROMPT},
            ]}], temperature=0.0, max_tokens=500)
            raw = raw.strip()
            if raw.startswith("```"): raw = raw.split("\n", 1)[-1].replace("```", "").strip()
            review = json.loads(raw)
            print(f"\nSeverity: {review.get('severity', 'N/A')}")
            print(f"Patterns: {review.get('patterns', [])}")
            print(f"Vision fix: {review.get('vision_prompt_fix', 'N/A')}")
            print(f"Code fix:   {review.get('code_system_fix', 'N/A')}")
        except Exception as e:
            print(f"Review failed: {e}")

    # --- Train-mode: aggregate code-review report ---
    if args.train:
        code_reviews = [s["code_review"] for s in results if s["code_review"] and "error" not in s["code_review"]]
        if code_reviews:
            print("\n=== Structural Code Review Summary ===")
            for s in results:
                cr = s["code_review"]
                if cr and "error" not in cr:
                    cs = cr.get("code_score", 0)
                    cd = cr.get("code_diagnosis", "")
                    print(f"  {os.path.basename(s['png'])}: code_score={cs:.1f} | {cd[:100]}")
            avg_code_score = sum(cr["code_score"] for cr in code_reviews) / len(code_reviews)
            print(f"\n  Avg code score: {avg_code_score:.2f} / 5.0")
            hacks = []
            for cr in code_reviews:
                hacks.extend(cr.get("structural_hacks", []))
            if hacks:
                print(f"  Structural hacks found: {len(hacks)}")
                for h in hacks[:5]:
                    print(f"    - {h}")


if __name__ == "__main__":
    main()

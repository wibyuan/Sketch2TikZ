"""
Batch benchmark: N samples per difficulty, internal critic only.
Usage: python batch_benchmark.py <difficulty> <num_samples>
Saves results to benchmark_results_<difficulty>.json
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(override=True)

from train.pipeline import generate, _internal_critic

def run_batch(difficulty: str, num: int = 10):
    out_dir = f"output_{difficulty}"
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for idx in range(num):
        png = f"test/data/{difficulty}/{idx:04d}.png"
        if not os.path.exists(png):
            continue
        print(f"[{difficulty}] START {idx:04d}.png", flush=True)
        t0 = time.time()
        try:
            r = generate(png, idx, output_dir=out_dir)
        except Exception as e:
            print(f"[{difficulty}] ERROR {idx:04d}.png: {type(e).__name__}: {str(e)[:80]}", flush=True)
            results.append({"idx": idx, "error": str(e)[:200], "score": 0.0, "compile_ok": False})
            continue
        elapsed = time.time() - t0
        c = _internal_critic(png, r.gen_pdf_path, out_dir) if r.compile_ok else None
        score = c["score"] if c else 0.0
        diag = c.get("diagnosis", "") if c else ""
        results.append({
            "idx": idx,
            "compile_ok": r.compile_ok,
            "compile_attempts": r.compile_attempts,
            "score": score,
            "diagnosis": diag,
            "elapsed": round(elapsed, 1),
        })
        print(f"[{difficulty}] DONE {idx:04d}.png | Compile={r.compile_ok} Score={score:.1f} Time={elapsed:.0f}s", flush=True)
    return results

def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "easy"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"\n=== {difficulty.upper()} ===", flush=True)
    results = run_batch(difficulty, num)
    with open(f"benchmark_results_{difficulty}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: benchmark_results_{difficulty}.json", flush=True)

    n = len(results)
    comp = sum(1 for x in results if x.get("compile_ok"))
    scores = [x["score"] for x in results if x.get("compile_ok")]
    avg = sum(scores) / len(scores) if scores else 0.0
    passes = sum(1 for x in results if x["score"] >= 3.0)
    print(f"{difficulty:10s} n={n} compile={comp}/{n} avg_score={avg:.2f} pass_rate={passes}/{n}", flush=True)

if __name__ == "__main__":
    main()

"""
Wrapped pipeline with progress callbacks for the web UI.
Does NOT modify train/pipeline.py — reuses its helpers and logic.
"""
import os, time, json
from typing import Callable, Optional

from train.pipeline import (
    VISION_PROMPT, CODE_SYSTEM, CRITIC_PROMPT,
    _fix, _clean, _compile, _pdf_to_png, _encode_img,
    _internal_critic as _orig_internal_critic,
)
from train.llm_caller import image_to_text, text_to_text, _create, VISION_MODELS, VISION_PLATFORMS, CODE_PLATFORMS


# ── Progress-aware generation ──────────────────────────────────

def generate_with_callbacks(
    image_path: str,
    output_dir: str,
    callbacks: Optional[Callable] = None,
    custom_prompt: Optional[str] = None,
    task_id: str = "",
) -> dict:
    """
    Run the full Sketch2TikZ pipeline with optional progress callbacks.

    callbacks(stage: str, status: str, message: str, data: dict = None)
        stage   : "vision" | "codegen" | "compile" | "critic" | "done" | "error"
        status  : "running" | "retry" | "success" | "fail"
        message : human-readable description
        data    : optional extra info (attempt number, score, etc.)

    custom_prompt: if provided, skips the vision model and uses this directly.
    """

    def _cb(stage, status, message, data=None):
        if callbacks:
            try:
                callbacks(stage, status, message, data or {})
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # ── Stage 1: Vision description ──
    if custom_prompt is not None:
        desc = custom_prompt
        _cb("vision", "success", "Using custom prompt (skipped auto description)", {"custom": True})
    else:
        _cb("vision", "running", "Analyzing sketch with vision model...")
        try:
            desc = image_to_text(image_path, VISION_PROMPT,
                                 platforms=VISION_PLATFORMS, temperature=0.0, max_tokens=1024)
            _cb("vision", "success", "Vision analysis complete", {"description": desc[:200]})
        except Exception as e:
            _cb("vision", "fail", f"Vision model failed: {e}")
            return {"error": f"Vision failed: {e}"}

    vision_time = round(time.time() - t_start, 1)

    tex_path = os.path.join(output_dir, "output.tex")
    pdf_path = os.path.join(output_dir, "output.pdf")
    png_path = os.path.join(output_dir, "output.png")

    msgs = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": f"Generate TikZ code for:\n{desc}"},
    ]

    # ── Stage 2: Code generation + compile self-heal ──
    t_code = time.time()
    compile_ok = False
    compile_attempts = 0
    critic_first_score = 0.0
    critic_final_score = 0.0
    diagnosis = ""
    tikz = ""

    for attempt in range(3):
        compile_attempts = attempt + 1
        _cb("codegen", "running", f"Generating TikZ code (attempt {compile_attempts}/3)...",
            {"attempt": compile_attempts})

        if attempt == 0 and custom_prompt is None:
            # First attempt: let code model see original image too
            code_prompt = CODE_SYSTEM + "\n\nGenerate TikZ code based on this description AND the original image:\n" + desc
            try:
                raw = image_to_text(image_path, code_prompt,
                                    platforms=[p for p in CODE_PLATFORMS if p in VISION_MODELS],
                                    temperature=0.0, max_tokens=4096)
            except Exception as e:
                _cb("codegen", "fail", f"Code generation failed: {e}", {"attempt": compile_attempts})
                return {"error": f"Code generation failed: {e}"}
        else:
            try:
                raw = text_to_text(msgs, platforms=CODE_PLATFORMS,
                                   temperature=0.0, max_tokens=4096)
            except Exception as e:
                _cb("codegen", "fail", f"Code generation failed: {e}", {"attempt": compile_attempts})
                return {"error": f"Code generation failed: {e}"}

        tikz = _clean(raw)
        tikz = _fix(tikz)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tikz)

        _cb("compile", "running", f"Compiling LaTeX (attempt {compile_attempts}/3)...",
            {"attempt": compile_attempts})
        ok, errors = _compile(tex_path, pdf_path)
        if ok:
            compile_ok = True
            _cb("compile", "success", "LaTeX compiled successfully",
                {"attempt": compile_attempts})
            break
        else:
            _cb("compile", "retry", f"Compile error on attempt {compile_attempts}: {errors[:150]}",
                {"attempt": compile_attempts, "errors": errors})
            msgs.append({"role": "user",
                         "content": f"Compile errors:\n{errors}\nFix and output complete code."})
    else:
        compile_ok = False
        _cb("compile", "fail", "All 3 compile attempts failed", {"attempts": 3})

    codegen_time = round(time.time() - t_code, 1)

    # ── Stage 3: Visual critic + self-heal ──
    if compile_ok:
        _cb("critic", "running", "Running visual quality review...")
        c1 = _orig_internal_critic(image_path, pdf_path, output_dir)
        critic_first_score = c1["score"]
        diagnosis = c1["diagnosis"]
        _cb("critic", "success" if c1["is_pass"] else "retry",
            f"Visual review score: {critic_first_score:.1f}/5.0 — {diagnosis[:200]}",
            {"score": critic_first_score, "is_pass": c1["is_pass"], "diagnosis": diagnosis})

        if not c1["is_pass"]:
            _cb("codegen", "running", "Applying visual self-heal fixes...")
            msgs.append({"role": "user",
                         "content": f"Visual review found these differences from the reference:\n"
                                    f"{diagnosis}\n\nMake ONLY minimal targeted fixes to address these "
                                    f"specific issues. Do NOT change anything that is already correct."})
            try:
                raw2 = text_to_text(msgs, platforms=CODE_PLATFORMS,
                                    temperature=0.0, max_tokens=4096)
                tikz2 = _clean(raw2)
                tikz2 = _fix(tikz2)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz2)
                ok2, _ = _compile(tex_path, pdf_path)
                if ok2:
                    tikz = tikz2
                    c2 = _orig_internal_critic(image_path, pdf_path, output_dir)
                    critic_final_score = c2["score"]
                    diagnosis = c2["diagnosis"]
                    _cb("critic", "success" if c2["is_pass"] else "fail",
                        f"Self-heal score: {critic_final_score:.1f}/5.0 — {diagnosis[:200]}",
                        {"score": critic_final_score, "is_pass": c2["is_pass"], "diagnosis": diagnosis})
                else:
                    critic_final_score = 0.0
                    _cb("compile", "fail", "Self-heal compile failed")
            except Exception as e:
                _cb("codegen", "fail", f"Self-heal failed: {e}")
        else:
            critic_final_score = c1["score"]
    else:
        _cb("critic", "fail", "Skipped visual review (compile failed)")

    # Render PNG preview
    if compile_ok and os.path.exists(pdf_path):
        _pdf_to_png(pdf_path, png_path)

    total_time = round(time.time() - t_start, 1)

    result = {
        "task_id": task_id,
        "compile_ok": compile_ok,
        "compile_attempts": compile_attempts,
        "vision_time": vision_time,
        "codegen_time": codegen_time,
        "total_time": total_time,
        "critic_first_score": critic_first_score,
        "critic_final_score": critic_final_score,
        "critic_pass": critic_final_score >= 3.0,
        "diagnosis": diagnosis,
        "description": desc,
        "tikz_code": tikz if compile_ok else "",
        "paths": {
            "tex": tex_path if compile_ok else "",
            "pdf": pdf_path if compile_ok else "",
            "png": png_path if os.path.exists(png_path) else "",
        },
    }

    _cb("done", "success" if compile_ok else "fail",
        f"Generation complete in {total_time}s" if compile_ok else "Generation failed",
        result)
    return result


# ── Direct recompile (no AI call) ──────────────────────────────

def compile_tex_only(tikz_code: str, output_dir: str) -> dict:
    """Save TikZ code and compile it directly. Returns compilation result."""
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "output.tex")
    pdf_path = os.path.join(output_dir, "output.pdf")
    png_path = os.path.join(output_dir, "output.png")

    tikz = _fix(_clean(tikz_code))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tikz)

    ok, errors = _compile(tex_path, pdf_path)
    if ok:
        _pdf_to_png(pdf_path, png_path)

    return {
        "compile_ok": ok,
        "errors": errors,
        "tex_path": tex_path,
        "pdf_path": pdf_path if ok else "",
        "png_path": png_path if ok else "",
    }

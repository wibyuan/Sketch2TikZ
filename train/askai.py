"""
Academic AI Review — ask vision model to analyse gaps between
ground-truth and generated TikZ, then propose concrete prompt fixes.
"""
import os, base64, json, textwrap, subprocess, shutil

from dotenv import load_dotenv
load_dotenv(override=True)

from train.llm_caller import _create, VISION_MODELS, VISION_PLATFORMS


def _gs() -> str:
    for name in ["gs", "gswin64c", "gswin64"]:
        f = shutil.which(name)
        if f:
            return f
    root = os.path.dirname(os.path.dirname(os.getenv("CONDA_PREFIX", "")))
    for sub in ["Library/bin/gs.exe", "Library/bin/gswin64c.exe"]:
        p = os.path.join(root, sub)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Ghostscript not found")


def _pdf_to_png(pdf_path: str, png_path: str) -> bool:
    try:
        subprocess.run(
            [_gs(), "-dNOPAUSE", "-dBATCH", "-dSAFER",
             "-sDEVICE=png16m", "-r150", "-dFirstPage=1", "-dLastPage=1",
             f"-sOutputFile={png_path}", pdf_path],
            capture_output=True, text=True, timeout=30,
        )
        return os.path.exists(png_path) and os.path.getsize(png_path) > 0
    except Exception:
        return False


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    return f"data:{'image/png' if ext == '.png' else 'image/jpeg'};base64,{data}"


# ── Prompts ──────────────────────────────────────────

REVIEW_SYSTEM = (
    "You are a senior computer-vision + graphics researcher reviewing a "
    "sketch-to-TikZ generation pipeline. Be rigorous, specific, and actionable."
)

REVIEW_USER_TEMPLATE = """I need an academic review of a sketch-to-TikZ pipeline failure.

## Inputs
1. REFERENCE image (hand-drawn sketch)
2. GENERATED image (rendered PDF from model output)
3. Ground-Truth TikZ code (human-written, correct)
4. Generated TikZ code (model output)
5. Critic diagnosis (automated visual comparison)

## Ground-Truth TikZ
```latex
{gt_code}
```

## Generated TikZ
```latex
{gen_code}
```

## Critic Diagnosis (score {score}/5.0)
{diagnosis}

## Task
Analyse the ROOT CAUSE of the failure by comparing GT vs generated code AND the visual images.

For each issue, determine whether it originates in:
- **Vision**: the image-description prompt failed to elicit a key detail
- **Code**: the code-generation system prompt lacked a rule or example
- **Model**: the LLM itself ignored an existing rule (rare, note if so)

Output ONLY a JSON object:
{{
  "root_cause": "<one-line summary of the dominant failure mode>",
  "vision_prompt_fix": "<concrete sentence to ADD to the vision prompt, or empty if none>",
  "code_system_fix": "<concrete rule to ADD to the code system prompt, or empty if none>",
  "example_fix": "<concrete mini-example to ADD to the code prompt, or empty>",
  "severity": "<critical|major|minor>",
  "confidence": "<high|medium|low>",
  "rationale": "<2-3 sentences explaining why this fix will help>"
}}

Be SPECIFIC. No vague advice. Every fix must be copy-pasteable."""


def review(
    original_png: str,
    gen_pdf: str,
    gt_code: str,
    gen_code: str,
    critic_score: float,
    critic_diagnosis: str,
    out_dir: str = "output",
) -> dict:
    """
    Ask the vision model to review a single failure case.
    Returns parsed JSON dict or error dict.
    """
    # Render PDF to PNG for side-by-side comparison
    gen_png = os.path.join(out_dir, "askai_gen.png")
    if not _pdf_to_png(gen_pdf, gen_png):
        return {"error": "PDF render failed for AI review"}

    b64_ref = _encode(original_png)
    b64_gen = _encode(gen_png)

    user_text = REVIEW_USER_TEMPLATE.format(
        gt_code=textwrap.shorten(gt_code, width=3000, placeholder="\n... (truncated)\n"),
        gen_code=textwrap.shorten(gen_code, width=3000, placeholder="\n... (truncated)\n"),
        score=critic_score,
        diagnosis=critic_diagnosis,
    )

    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "REFERENCE sketch:"},
            {"type": "image_url", "image_url": {"url": b64_ref}},
            {"type": "text", "text": "GENERATED output:"},
            {"type": "image_url", "image_url": {"url": b64_gen}},
            {"type": "text", "text": user_text},
        ]},
    ]

    raw = None
    for p in VISION_PLATFORMS:
        try:
            raw = _create(p, VISION_MODELS[p], messages, temperature=0.0, max_tokens=800)
            break
        except Exception:
            continue

    if raw is None:
        return {"error": "All vision platforms failed for AI review"}

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].replace("```", "").strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed: {raw[:200]}", "raw": raw}

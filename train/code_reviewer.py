"""
Code-level reviewer — structural comparison of generated TikZ vs ground-truth.

Architecture:
  Layer 1: Programmatic checks (regex/AST-like) — objective, explainable
  Layer 2: LLM interpretation — explains Layer-1 findings + catches semantic gaps

Run ONLY on train data (which has GT code). This is a diagnostic tool, not
feeding answers to the model during inference.
"""
import os, re, json, textwrap
from difflib import unified_diff

from dotenv import load_dotenv
load_dotenv(override=True)

from train.llm_caller import text_to_text, CODE_PLATFORMS


# ── Layer 1: Programmatic checks ───────────────────

_TIKZ_COMMANDS = [r"\\node", r"\\draw", r"\\fill", r"\\path", r"\\coordinate",
                  r"\\rectangle", r"\\circle", r"\\arc", r"\\line"]


def _count_commands(code: str) -> dict:
    """Count occurrences of common TikZ commands."""
    return {cmd: len(re.findall(cmd + r"(?![a-zA-Z])", code)) for cmd in _TIKZ_COMMANDS}


def _extract_packages(code: str) -> set:
    """Extract \\usepackage{...} names."""
    return set(re.findall(r"\\usepackage\[(?:[^\]]*)\]?\{([^}]+)\}", code))


def _extract_coordinates(code: str) -> list:
    """Extract numeric coordinates like (1.2, 3.4)."""
    return re.findall(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", code)


def _extract_node_labels(code: str) -> set:
    """Extract node text labels like \\node ... {$x_1$};"""
    return set(re.findall(r"\\node.*?\{([^}]*)\}", code))


def _has_documentclass(code: str) -> bool:
    return r"\documentclass" in code


def _has_tikzpicture(code: str) -> bool:
    return r"\begin{tikzpicture}" in code


def programmatic_review(gt_code: str, gen_code: str) -> dict:
    """
    Objective, rule-based comparison. No AI involved.
    Returns structured findings for LLM layer to interpret.
    """
    gt_counts = _count_commands(gt_code)
    gen_counts = _count_commands(gen_code)

    command_diffs = {}
    for cmd in _TIKZ_COMMANDS:
        d = gen_counts.get(cmd, 0) - gt_counts.get(cmd, 0)
        if d != 0:
            command_diffs[cmd] = d

    gt_pkgs = _extract_packages(gt_code)
    gen_pkgs = _extract_packages(gen_code)
    missing_pkgs = gt_pkgs - gen_pkgs
    extra_pkgs = gen_pkgs - gt_pkgs

    gt_coords = _extract_coordinates(gt_code)
    gen_coords = _extract_coordinates(gen_code)

    # Simple coordinate-range comparison
    gt_x = [float(c[0]) for c in gt_coords if c[0]]
    gt_y = [float(c[1]) for c in gt_coords if c[1]]
    gen_x = [float(c[0]) for c in gen_coords if c[0]]
    gen_y = [float(c[1]) for c in gen_coords if c[1]]

    coord_range_issue = False
    if gt_x and gen_x and (max(gen_x) - min(gen_x)) > 2 * (max(gt_x) - min(gt_x)):
        coord_range_issue = True
    if gt_y and gen_y and (max(gen_y) - min(gen_y)) > 2 * (max(gt_y) - min(gt_y)):
        coord_range_issue = True

    structural_checks = {
        "has_documentclass": _has_documentclass(gen_code),
        "has_tikzpicture": _has_tikzpicture(gen_code),
        "gt_lines": len(gt_code.splitlines()),
        "gen_lines": len(gen_code.splitlines()),
        "command_diffs": command_diffs,
        "missing_packages": list(missing_pkgs),
        "extra_packages": list(extra_pkgs),
        "coordinate_count_diff": len(gen_coords) - len(gt_coords),
        "coordinate_range_issue": coord_range_issue,
    }

    # Simple heuristic score (0-5) based on objective gaps
    deductions = 0.0
    if not structural_checks["has_documentclass"]:
        deductions += 2.0
    if not structural_checks["has_tikzpicture"]:
        deductions += 2.0
    deductions += min(1.0, len(command_diffs) * 0.2)
    deductions += min(1.0, len(missing_pkgs) * 0.3)
    if coord_range_issue:
        deductions += 1.0

    structural_checks["heuristic_score"] = max(0.0, 5.0 - deductions)
    return structural_checks


# ── Layer 2: LLM interpretation ────────────────────

CODE_REVIEW_SYSTEM = (
    "You are a TikZ code-structure auditor. You are given objective, "
    "programmatic findings (command counts, package diffs, coordinate ranges) "
    "plus a unified diff of GT vs generated code. Explain the findings and "
    "identify any additional semantic issues the programmatic layer missed."
)

CODE_REVIEW_TEMPLATE = """## Objective Findings (programmatic layer)
{findings}

## Unified Diff (GT → Generated)
```diff
{diff}
```

## Task
Based on the objective findings above, provide:
1. An interpretation of what the programmatic differences mean structurally
2. Any semantic issues the regex-based layer likely missed (e.g., correct primitive but wrong parameter)
3. Whether the structural hacks are benign workarounds or fundamental errors

Output ONLY JSON:
{{
  "code_score": <float 0.0-5.0, should align with heuristic but can override if semantic analysis justifies>,
  "is_pass": <true/false>,
  "code_diagnosis": "<one sentence summary>",
  "programmatic_interpretation": "<2-3 sentences explaining what the command/package/coordinate diffs mean>",
  "missed_by_regex": ["<semantic issues the regex layer likely missed>"],
  "severity": "low|medium|high"
}}"""


def review_code(gt_code: str, gen_code: str) -> dict:
    """
    Two-layer review:
      1. Programmatic checks (objective, no AI)
      2. LLM interpretation of findings + semantic gap analysis
    """
    # Layer 1
    findings = programmatic_review(gt_code, gen_code)

    # Unified diff for LLM context
    gt_lines = gt_code.splitlines(keepends=True)
    gen_lines = gen_code.splitlines(keepends=True)
    diff = "".join(unified_diff(gt_lines, gen_lines, fromfile="GT", tofile="GEN", lineterm=""))
    if len(diff) > 6000:
        diff = diff[:3000] + "\n... [truncated] ...\n" + diff[-3000:]

    # Layer 2
    gt_trim = textwrap.shorten(gt_code, width=2000, placeholder="\n... (truncated)\n")
    gen_trim = textwrap.shorten(gen_code, width=2000, placeholder="\n... (truncated)\n")

    findings_text = json.dumps(findings, indent=2, ensure_ascii=False)
    user_text = CODE_REVIEW_TEMPLATE.format(findings=findings_text, diff=diff)

    messages = [
        {"role": "system", "content": CODE_REVIEW_SYSTEM},
        {"role": "user", "content": user_text},
    ]

    raw = None
    for p in CODE_PLATFORMS:
        try:
            raw = text_to_text(messages, platforms=[p], temperature=0.0, max_tokens=600)
            break
        except Exception:
            continue

    if raw is None:
        # Fallback: return only programmatic layer if LLM fails
        return {
            "code_score": findings["heuristic_score"],
            "is_pass": findings["heuristic_score"] >= 3.0,
            "code_diagnosis": "Programmatic review only — LLM layer failed.",
            "programmatic_findings": findings,
            "severity": "medium" if findings["heuristic_score"] < 3.0 else "low",
        }

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].replace("```", "").strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    try:
        llm_result = json.loads(raw)
        # Merge programmatic findings into result for transparency
        llm_result["programmatic_findings"] = findings
        return llm_result
    except json.JSONDecodeError:
        return {
            "code_score": findings["heuristic_score"],
            "is_pass": findings["heuristic_score"] >= 3.0,
            "code_diagnosis": f"LLM parse failed. Raw: {raw[:200]}",
            "programmatic_findings": findings,
            "severity": "medium" if findings["heuristic_score"] < 3.0 else "low",
        }

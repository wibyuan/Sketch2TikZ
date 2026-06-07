# Sketch2TikZ — Agent 自我迭代指南

> 本文件面向 coding agent。如果你被调用来改进这个项目的 TikZ 生成质量，请按此流程执行。

## 1. 项目目标

将手绘草图（PNG）自动转换为可编译的 TikZ 矢量图代码（LaTeX），最终渲染为 PDF。

核心指标：
- **Compile Rate** — 生成代码能否通过 XeLaTeX 编译
- **Fidelity Score** — 视觉相似度（1.0~5.0，由 sealed judge 评定）
- **Pass Rate** — score ≥ 3.0 的样本比例

## 2. 你能修改什么

可自由修改（在 `train/` 下）：
- `train/prompts.py` — **Prompt 仓库**，迭代优化的主战场
- `train/pipeline.py` — 生成流程、后处理逻辑、编译修复
- `train/dev_loop.py` — 迭代循环脚本本身
- `train/askai.py` — AI Review 的 prompt 和调用逻辑

**禁止修改**（sealed）：
- `test/judge.py` — 最终评分器，只读
- `test/runner.py` — 评测入口，只读
- `test/data/` — 测试集，无 ground truth 代码

## 3. 自我迭代工作流（标准三步）

### Step 1：在 train 数据上诊断弱点

```bash
# 跑 easy 难度的 train 数据，启用 AI review（自动分析低分样本）
python -m train.dev_loop --difficulty easy --samples 5 --review

# 如果 easy 表现已经很好，换 medium / difficult
python -m train.dev_loop --difficulty medium --samples 5 --review
```

**输出位置**：`iter_runs/{difficulty}_{timestamp}/`
- `results.json` — 每个样本的详细结果（compile、score、diagnosis）
- `report.md` — AI 汇总的改进建议（按 severity 分类，已去重）
- `prompts_snapshot.json` — 当前使用的 prompt 备份

**关注重点**：
- 哪些样本 score < 3.0？
- `report.md` 里的 `root_cause` 是 Vision 问题还是 Code 问题？
- `vision_prompt_fix` 和 `code_system_fix` 是否具体、可粘贴？

**新增：代码结构审查（train 数据专用）**

Train 数据包含 ground-truth TikZ 代码。除视觉对比外，还应做**纯代码层面的结构审查**——检查生成代码与 GT 在 TikZ 原语、坐标、样式上是否一致。这可以发现视觉对比漏掉的"结构性 hack"（如画得对但用了错误 primitive）。

```bash
# 在 train 数据上同时跑视觉 critic + 代码结构审查
python -m train.review_test --train --difficulty easy --num-samples 5
```

**重要边界**：代码审查是**诊断工具**，不是给模型喂答案。`test/` 数据仍然无 GT 代码，最终评测完全密封。

### Step 2：应用建议并修改 Prompt

根据 `report.md` 的 **Recommended Prompt Changes**，编辑 `train/prompts.py`：

```python
# VISION_PROMPT — 控制模型如何描述图片
VISION_PROMPT = (
    "..."
)

# CODE_SYSTEM — 控制模型如何生成 TikZ 代码
CODE_SYSTEM = (
    "..."
)

# CATEGORY_PROMPTS — 按内容类型定制的专用 prompt（可选）
CATEGORY_PROMPTS = {
    "chart_plot": {...},
    "math_formula": {...},
    "math_geometry": {...},
    "pure_drawing": {...},
}
```

**修改原则**：
- 一次只加 1~2 条具体规则，不要堆砌
- 新增规则必须是 **copy-pasteable 的句子**，不能是模糊建议
- 优先修改导致最多样本失败的 root cause
- 如果分类 prompt 导致某类数据变差，回退到通用 prompt 或精简该类 prompt

### Step 3：验证改进效果

```bash
# 用相同样本重新跑，对比前后分数
python -m train.dev_loop --difficulty easy --samples 5 --review
```

对比方式：
1. 打开上一次和这一次的 `results.json`
2. 比较同一 `basename` 的 `critic_score`
3. 确认 compile_rate 没有下降
4. 如果 avg_score 提升且 pass_rate 提升 → 改进有效

**A/B 对比快捷命令**（针对分类数据）：
```bash
# 在 content_category_v2 的 sample 上对比通用 vs 分类 prompt
python -m train.category_test chart_plot
python -m train.category_test math_formula
python -m train.category_test math_geometry
python -m train.category_test pure_drawing
```

## 4. 最终验证（sealed test）

Prompt 调好后，必须在 **test 数据**（无 GT 代码）上验证，防止过拟合 train：

```bash
python -m test.runner --difficulty easy --num-samples 10
```

这条命令会：
1. 读取 `test/data/easy/` 的 PNG（模型从未见过）
2. 调用 `train/pipeline.py` 生成 TikZ
3. 用 sealed judge 评分
4. 输出 Compile Rate / Pass Rate / Avg Score

**这是唯一可信的指标**。如果 train 上提升但 test 上下降 → 过拟合，回退修改。

## 5. 快速质检（单张图调试）

如果想快速看某一张图的效果，不用跑完整 dev_loop：

```bash
python -c "
from train.pipeline import generate
r = generate('path/to/image.png', 0, output_dir='output')
print('Compile:', r.compile_ok, 'Attempts:', r.compile_attempts)
"
```

生成产物在 `output/`：
- `gen_0000.tex` — 生成的 TikZ 代码
- `gen_0000.pdf` — 编译后的 PDF
- `gen_0000.log` — XeLaTeX 编译日志（编译失败时看这里）
- `critic_internal.png` — 内部 critic 对比用的渲染图

## 6. 常见陷阱

| 陷阱 | 表现 | 对策 |
|------|------|------|
| Prompt 过度指定 | 模型按错误理解硬套规则，score 下降 | 精简 prompt，删除抽象概念，保留具体例子 |
| 编译失败率上升 | 新增规则引入不兼容宏包或语法 | 检查 `_fix()` 和后处理，确保兼容 xelatex |
| Train 提升 Test 下降 | 过拟合 train 数据的特定模式 | 必须在 test 数据上验证，必要时回退 |
| AI review 建议太泛 | `vision_prompt_fix` 是空话 | 拒绝采纳，要求更具体的规则或示例 |
| Critic 评分波动 | 同一图两次评分不同 | 正常现象，看趋势不看单点；多跑几个样本取平均 |

## 7. 迭代 checklist

每次提交前确认：
- [ ] `train/prompts.py` 的修改有明确依据（来自 report.md 或 A/B 测试）
- [ ] `python -m test.runner --difficulty easy --num-samples 10` 通过
- [ ] Compile Rate ≥ 80%，Avg Score ≥ 3.0（easy 难度的基线）
- [ ] `iter_runs/` 和 `cat_bench/` 的临时文件已清理（`git status` 确认）
- [ ] `AGENTS.md` 已更新（如果迭代流程本身有变化）

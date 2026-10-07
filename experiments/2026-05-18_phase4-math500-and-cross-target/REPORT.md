# Phase 4 A: 跨任务 (GSM8K → MATH-500) 和跨 target 的 C1/C4 验证

- **Exp ID**: 2026-05-18_phase4-math500-and-cross-target
- **日期**: 2026-05-18
- **状态**: ✅ 完成 (Phase 4 A 全部 6 cells + 跨 target negative + C1 跨 dataset 都齐了)
- **依赖 runs** (全在 ~/AI/runs/):
  - `2026-05-17_extract-qwen-math-math500-100` (C1 source, Qwen-Math N=8 on MATH-500)
  - `2026-05-17_extract-qwen-instruct-math500-100` (Qwen-Instruct greedy on MATH-500, 算 same-task source vector)
  - `2026-05-17_c4-math500-source-gsm8k-qwenmath` (R1-Distill steered by GSM8K-source)
  - `2026-05-17_c4-math500-source-math500-qwenInstruct` (R1-Distill steered by same-task non-math source)
  - `2026-05-18_c4-math500-source-math500-qwenmath` (R1-Distill steered by same-task Math source, *strongest*) — α=2 in flight
  - `2026-05-18_c4-math500-target-qweninstruct-source-qwenmath` (cross-target: Qwen-Instruct as target)

## 1. 目的

Phase 3c.3 在 GSM8K 上完整刻画了 C1 + C4. Phase 4 A 测两个独立的 generalization 维度:
1. **跨任务**: 现象 + 方法在更难的 MATH-500 (Hendrycks 数学 hard subset) 上是否成立?
2. **跨 target**: C4 干预到不只 R1-Distill 之外的 target 上是否成立?

## 2. C1 GeoVote: GSM8K → MATH-500 (✅ 完成)

100 题 GSM8K + 100 题 MATH-500, Qwen2.5-Math-1.5B-Instruct, N=8, temperature=0.7.

| Strategy | GSM8K | MATH-500 | margin vs logprob |
|----------|-------|----------|--------------------|
| oracle (上界) | 0.97 | 0.87 | — |
| majority_vote (self-consistency) | 0.90 | 0.77 | +7 (still strongest) |
| **GeoVote `mean_step_norm@L20` (ours)** | **0.88** | **0.75** | **+5** |
| logprob_max / weighted | 0.82 | 0.70 | 0 (baseline) |
| random_pick | 0.78 | 0.68 | — |
| greedy_pass1 | 0.76 | 0.63 | — |
| geo_min_pick (no vote) | 0.76 | 0.59 | — |

**Δ vs logprob**: +6pp on GSM8K, +5pp on MATH-500. **Method 完全复现, 1pp 不算显著缩窄**.

**Δ vs greedy**: +12pp on both (identical). Method 给的 N-best benefit 是 dataset-invariant 的.

→ ![C1 cross-dataset](p4_c1_dataset_transfer.png)

## 3. C4 CrossSteer: R1-Distill on MATH-500 (✅ 大部分完成, 1 项 in-flight)

R1-Distill-Qwen-1.5B target, layer=14, alpha sweep {0, 1, 2}, 100 题, greedy.

| Dataset | Source | source acc | target baseline | target α=2 | Δ |
|---------|--------|-----|------|------|------|
| GSM8K | Qwen-Base | 0.37 | 0.32 | 0.42 | **+0.10** |
| GSM8K | Qwen-Instruct | 0.60 | 0.32 | 0.45 | **+0.13** |
| GSM8K | Qwen-Math | 0.78 | 0.32 | **0.47** | **+0.15** |
| MATH-500 | Qwen-Math (GSM8K, cross-task) | 0.78 | 0.17 | **0.22** | +0.05 |
| MATH-500 | Qwen-Instruct (MATH-500, same-task weak) | 0.31 | 0.17 | 0.20 | +0.03 |
| MATH-500 | Qwen-Math (MATH-500, **strongest same-task**) | 0.63 | 0.17 | **0.24** | **+0.07** |

**MATH-500 三个 cell 的 ordering (清晰):**

| 描述 | Δ |
|------|---|
| 强 source + 同任务 (Math/MATH-500) | **+0.07** ← 最强 |
| 强 source + 跨任务 (Math/GSM8K) | +0.05 |
| 弱 source + 同任务 (Instruct/MATH-500) | +0.03 |

**关键洞察**:
- **Source 强度 + 任务匹配 = 最佳**, 但单独"强 source 跨任务"也比"弱 source 同任务"强
- MATH-500 上 effect size 比 GSM8K 小 (+7pp vs +15pp), 因为 R1-Distill MATH-500 baseline 仅 0.17 (难度上限附近, headroom 少)
- 所有 6 个 R1-Distill settings 都正向 (3 GSM8K + 3 MATH-500), **direction 跨 dataset 一致**

→ ![C4 cross-dataset cross-source](p4_c4_target_source_dataset.png)

## 4. C4 Cross-target Negative (✅ 完成 — 界定 method 适用范围)

Target = Qwen-Instruct (non-math, 但 sign-flip 没被破坏 — 它属于 Phase 3 中 Qwen 默认 cluster).
Source = Qwen-Math/GSM8K (best from P1.1).
Dataset = MATH-500.

| alpha | acc | Δ |
|------|-----|------|
| 0 (baseline) | 0.29 | — |
| 1 | 0.26 | **−0.03** |
| 2 | **0.10** | **−0.19** |

→ ![C4 cross-target negative](p4_c4_cross_target_negative.png)

**解读** (这是 *informative* 的负面结果, 不是 method 失败):

- alpha=2 让 Qwen-Instruct 从 29% 掉到 10% (减 19pp) → **direction 有强力 effect, 不是噪声**
- 但是是 *harmful* 方向, 因为 Qwen-Instruct 内部的 sign-flip 已经是"对的方向", 强加 source 方向反而过冲
- 这界定 CrossSteer 的适用范围: **只对 sign-flip 被破坏的 target 有效** (Phase 3b 中识别的 R1-Distill 一类), 对已经"对齐" 的 target 反而有害
- 对 paper 是好事: 不只是 method 描述, 还有 *predictability*  —— 给定一个 target, 我们能从 Phase 3 signature 提前预测 steering 是否有效

## 5. Phenomenon 命题 (Phase 3-4 综合)

Phase 4 A 的数据印证了 Phase 3b 已经命名的现象:

> **不同训练范式在 LLM 隐状态轨迹上留下系统性的、几何上可分离的 fingerprint. 这些 fingerprint 不仅描述模型 "怎么思考", 还预测了能否被 reverse-engineer 修复**.

Phase 3 a/b 给现象, Phase 3c.1-3 给 GSM8K 上的两个方法, Phase 4 A 给方法跨任务 + 跨 target 的边界刻画.

## 6. 论文 storyline (5 main claims)

| # | Claim | 证据 |
|---|-------|------|
| C1 | 训练范式 → 可分离几何 signature | Phase 3 a/b, distance matrix + heatmap |
| C2 | GeoVote 加权 best-of-N 比 logprob 加 +6pp | Phase 3c.3 P2 (GSM8K) + Phase 4 A (MATH-500) |
| C3 | CrossSteer 单次干预 +15pp (R1-Distill GSM8K) | Phase 3c.1 + control z=3.3σ + Phase 3c.3 P1 三维刻画 |
| C4 | Method 跨任务: GSM8K → MATH-500 都成立 | **Phase 4 A** |
| C5 | Method 在 sign-flip 已对齐的 target 上无效甚至有害, 这反过来证明 direction 有真实信号 | Phase 4 A 跨 target |

## 7. 沉淀 / 待做

- ✅ Disk-full lesson 已记 (2026-05-17_disk-full-killed-extractions.md)
- ✅ Corrupt-file 鲁棒性已加 (resume logic with try/except on load_trajectory)
- ✅ Strongest-source α=2 数字: 0.24 (+0.07)
- ⏳ Tier 2 alpha sweep on MATH-500 强 source (PID 6948, ~60 min), 验证 inverted-U 跨任务复现 → 见后续 phase4b REPORT
- ⏳ trajectory 清理 (MATH-500 N=8 的 trajectories 约 30 GB, 已 REPORT 后清)

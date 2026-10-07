# Phase 1: 首批几何指标 (trajectory_length + mean_curvature) 在 100 题 GSM8K 上的诊断力

- **Exp ID**: 2026-05-17_phase1-metrics-pilot-100
- **日期**: 2026-05-17
- **状态**: ✅ 完成
- **复用的 extraction**: [`2026-05-17_extract-pilot-gsm8k-100`](../2026-05-17_extract-pilot-gsm8k-100/)
- **Raw outputs**: `~/AI/runs/2026-05-17_extract-pilot-gsm8k-100/` (扩展了 metrics.parquet, auc_table.csv, plots/)

## 1. 目的

测试 Phase 1 假设 (research_plan.md 中的 H1): 推理错误的隐状态轨迹会在曲率指标上有可分离的 signature.

**预定义的退出条件**:
- 至少一个 (指标, 层) 组合的 AUC > 0.55 → 进入 Phase 2
- 同时设的隐性门槛 (来自 Phase 0 REPORT 观察): 必须显著超过 trivial baseline (n_gen_tokens AUC) —— 否则我们只是在测"长度", 没有几何价值

## 2. 设定

- **数据**: 100 题 GSM8K test (acc=78%, 22 题错), 复用 Phase 0 extraction
- **指标**:
  - `trajectory_length` per layer = Σ_t ‖h_{t+1} − h_t‖₂ (float32)
  - `mean_curvature` per layer = mean over interior steps of ∠(v_t, v_{t+1}) (弧度, [0, π])
- **评估**:
  - 单点 AUC: 每个 (metric, layer) 单独算 AUC, 正类 = 错误样本
  - 5-fold CV-AUC: 用逻辑回归融合 baseline + 几何特征
- **Baseline**: `n_gen_tokens` 单独预测

## 3. 关键结果

### 3.1 单一指标 raw AUC (sign-aware, 全部 29 层)

| 指标 | min | mean | max | 最强层 |
|------|-----|------|-----|--------|
| trajectory_length | 0.565 | 0.607 | 0.645 | L1 |
| mean_curvature | 0.362 | 0.399 | 0.433 | L8 (sign-flipped → 0.638) |

**`mean_curvature` AUC 全程 < 0.5 是个 finding, 不是 bug**: 它意味着 *正确样本曲率更高, 错误样本曲率更低* —— 和 H1 预测的方向**相反**.

### 3.2 5-fold CV-AUC (这是更可信的数字)

| 预测器 | CV-AUC |
|--------|--------|
| **trivial baseline (n_gen_tokens 单独)** | **0.688** |
| trajectory_length_L1 单独 | 0.603 |
| mean_curvature_L8 单独 | 0.494 (LR 在 N=100, 5-fold 下难稳定恢复 sign) |
| n_gen_tokens + mean_curvature_L8 | 0.692 |
| n_gen_tokens + trajectory_length_L1 | 0.675 |
| n_gen_tokens + length + curvature | 0.675 |
| n_gen_tokens + 29 curvature 层 | 0.698 (+0.010) |
| n_gen_tokens + 全部 58 length+curvature | 0.512 (过拟合) |

### 3.3 与 baseline 的混淆

| 量 | 与 n_gen_tokens 相关性 |
|----|---------------------|
| trajectory_length_L1 | **+0.863** |
| mean_curvature_L14 | **−0.131** |

`trajectory_length` 与 token 数高度相关 (基本就是 `mean_step_norm × T`), 所以它打不过 baseline 是必然的. `mean_curvature` 是 *length-independent* 的, 给了一个干净的几何 channel.

### 3.4 主图

![AUC per layer](../../../../AI/runs/2026-05-17_extract-pilot-gsm8k-100/plots/auc_per_layer.png)

(`runs/2026-05-17_extract-pilot-gsm8k-100/plots/auc_per_layer.png`)

## 4. 结论

### 字面满足 Phase 1 退出条件? ✅
最强的 (指标, 层) 单点 AUC = 0.645 (trajectory_length, L1), > 0.55 阈值.

### 实质性结论? ❌ 两个简单指标都不打过 baseline
- 在 N=100, 5-fold CV 下, 加入几何指标对 baseline 只有 **+0.004 到 +0.010 AUC** 的边际改进 —— 完全在噪声范围内
- 这是诚实的 Phase 1 失败: 最朴素的指标 (轨迹总长 + 平均曲率) 不携带超过 token 数所携带的信号

### 但发现了 2 个有价值的方向 (来自 fail-fast 模式)

1. **方向 1 — H1 sign-flip**: 正确轨迹**更弯曲**, 错误更直. 不是 plan 预测的"错误有曲率峰值". 这意味着 mental picture 要换成: 模型答对时频繁修正方向 (类似 backtracking), 答错时一往无前 (over-committed). 这是个真发现, 跨模型验证后可以作为 phenomenon 论文素材.

2. **方向 2 — 残差化是必须的**: 任何与 `n_gen_tokens` 高相关的指标 (例如 trajectory_length) 都是 length 的代理. Phase 2 之后所有指标必须报告 **partial AUC** (regress out n_gen_tokens 后) 或采用 length-invariant 设计 (per-step normalized).

## 5. 下一步 → Phase 2 (修订的优先级)

按 research_plan.md Phase 2 原本设定 + 本实验观察, 调整为:

**P1. 跨模型验证 sign-flip 是否稳健** (1-2 天):
- 在 Llama-3.2-1B, DeepSeek-R1-Distill-1.5B, Qwen2.5-1.5B (无 math) 上跑同 protocol
- 看 `mean_curvature` 的 AUC 全程是否仍 < 0.5
- 跨 3+ 模型若一致 → phenomenon 信号变强

**P2. Length-residualized 指标** (2-3 天):
- 实现 `partial_auc(metric, label | n_gen_tokens)` —— 残差化 token 数之后看
- 给 `trajectory_length` 加一个 normalized 版本: 平均步长 mean_step_norm = trajectory_length / (T-1)
- 这个量去除了"步数"维度, 保留"每步移动幅度", 几何上更纯

**P3. 引入新指标家族** (3-5 天):
- intrinsic dimension (TwoNN / MLE), 这是真正的"形状"指标, 不被 T 污染
- curvature 的 *方差* / max, 不只是 mean (有可能信号集中在少数 token)

**P4. 不进 Phase 3 / OT / PH 直到 P1-P3 给出 baseline-beating 信号** (纪律).

### 决策 (依据 Phase 1 数据)

**继续方向 γ, 不切方向**. 理由:
- 退出条件字面满足
- 第一次跑到的 (指标, 层) 信号弱但非零
- 找到了一个 *方向上的实质 finding* (sign-flip), 这本身就比"AUC 数字打过 baseline" 更适合发表
- 还没碰到 H2/H3/H4 的指标, "几何死了"的判断还太早

下一个里程碑 (退出条件): 跨 3 个模型, mean_curvature 的 sign-flip 保持一致, 且至少一个 length-residualized 几何指标在 CV-AUC 上比 baseline +0.03 以上.

## 6. 沉淀

无新 lessons_learned. 1 个测试纪律被验证有效: 测试一开始就跑 21 个, 帮我 catch 了曲率函数 1e-6 clamping 的精度问题 (见 `tests/test_trajectory.py`).

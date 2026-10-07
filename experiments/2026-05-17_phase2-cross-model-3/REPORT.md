# Phase 2 (P1 + P2): 跨模型 sign-flip 验证 + length-residualized 指标家族

- **Exp ID**: 2026-05-17_phase2-cross-model-3
- **日期**: 2026-05-17
- **状态**: ✅ 完成 (退出条件 ❌ 未达成, 但发现了 reframing 方向)
- **聚合 outputs**: `~/AI/runs/2026-05-17_phase2-cross-model-3/`
- **依赖 extractions**:
  - `2026-05-17_extract-pilot-gsm8k-100` (Qwen2.5-Math-1.5B-Instruct)
  - `2026-05-17_extract-deepseek-r1-distill-1.5b-100`
  - `2026-05-17_extract-qwen-1.5b-nomath-100`
- **被中止的 extraction**: `2026-05-17_extract-llama-1b-100` (Llama-3.2-1B, ModelScope 限速 660 kB/s, 见 [lesson](../../docs/lessons_learned/2026-05-17_modelscope-per-file-throttling.md))

## 1. 目的

按 [research_plan.md Phase 2](../../docs/research_plan.md) 的两条主线:
- **P1**: 在 ≥3 个不同模型上验证 Phase 1 找到的 `curvature_mean` sign-flip phenomenon 是否稳健
- **P2**: 引入 length-residualized 指标家族 (`mean_step_norm`, `curvature_var/max/p90/p99`, `partial_auc`), 摆脱 `n_gen_tokens` baseline 的混淆

**预定义的退出条件**:
- 至少 1 个 length-residualized 几何指标在 ≥3 个模型上 CV-AUC 比 n_gen_tokens baseline 高 ≥0.03
- 或: sign-flip phenomenon 在 ≥3 个模型上方向一致

## 2. 设定

| 项 | 值 |
|----|----|
| **模型 1** | Qwen/Qwen2.5-Math-1.5B-Instruct (SFT-only, math-tuned) |
| **模型 2** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (R1-RL distilled) |
| **模型 3** | Qwen/Qwen2.5-1.5B-Instruct (SFT-only, no math) |
| **数据集** | GSM8K test, 前 100 题 (固定, 跨模型一致) |
| **解码** | greedy, max_new_tokens=512 |
| **指标家族 (7 个)** | trajectory_length, mean_step_norm, curvature_{mean,var,max,p90,p99} |
| **评估方法** | per-(metric, layer) raw AUC + partial_auc (linear-residualize n_gen_tokens) |
| **判别极性** | "incorrect" 为正类, 故 AUC < 0.5 表示 sign-flipped (该指标越高 = 越正确) |

## 3. 主结果

### 3.1 三模型基线与精度

| 模型 | accuracy | n_gen_tokens AUC (baseline) |
|------|----------|------------------------------|
| Qwen2.5-Math-1.5B | 78% | 0.712 |
| DeepSeek-R1-Distill-1.5B | 32% | 0.588 |
| Qwen2.5-1.5B (no-math) | 60% | 0.721 |

精度差异巨大 (32%–78%). baseline AUC 反映"token 数对该模型有多大区分力" —— DeepSeek 因大多错, n_gen_tokens 区分力反而低.

### 3.2 各指标 best-layer 表现 (signed partial AUC)

| 指标 | Qwen-Math (b=0.712) | DeepSeek (b=0.588) | Qwen-nomath (b=0.721) | sign 一致? | 都打过 baseline? |
|------|---------------------|---------------------|------------------------|-----------|------------------|
| trajectory_length | 0.531 | **0.653** (+0.07) | 0.627 | ✓ 全不 flip | ✗ Qwen 两个没过 |
| **mean_step_norm** | **0.780** (+0.07) | **0.656** (+0.07) | 0.694 (−0.03) | ✗ Qwen 都 flip, DeepSeek 不 | ✗ Qwen-nomath 差一点 |
| curvature_mean | 0.534 | 0.565 | 0.558 | ✗ DeepSeek 不 flip | ✗ 全没过 |
| curvature_max | 0.661 | 0.634 (+0.05) | 0.542 | ✗ Qwen-nomath 不 flip | ✗ |
| curvature_p90 | 0.614 | 0.595 | 0.567 | ✗ | ✗ |
| curvature_p99 | 0.667 | 0.547 | 0.589 | ✗ | ✗ |
| curvature_var | 0.519 | 0.556 | 0.645 | ✓ 全不 flip | ✗ |

(`signed = max(auc, 1-auc)`, 即"无论方向哪边, 区分力的绝对值")

### 3.3 跨模型 curvature_mean 图

`~/AI/runs/2026-05-17_phase2-cross-model-3/plots/cross_model_curvature_per_layer.png`

(本地 mirror 在 `experiments/2026-05-17_phase2-cross-model-3/cross_model_curvature_per_layer.png`)

## 4. 关于退出条件: ❌ 未达成

| 退出条件 | 状态 |
|---------|------|
| 至少 1 个 length-residualized 指标在 ≥3 个模型上比 baseline +0.03 | **❌** `mean_step_norm` 只在 2/3 个模型上 (Qwen-nomath 上 −0.03) |
| `curvature_mean` sign-flip 在 ≥3 模型上一致 | **❌** Qwen 两个 flip, DeepSeek 不 flip |

字面上 Phase 2 P1+P2 **没有达到任何一个预定义退出条件**.

## 5. 但有 2 个真正有意思的发现

### Finding 1: `mean_step_norm` 是 **math-tuning 特异** 信号

- Qwen-Math (math-SFT) + DeepSeek (math-RL-distilled): 都 +0.07 partial AUC, 是当前所有指标最强
- Qwen-nomath (general SFT): −0.03 (反而低于 baseline)

→ 假设: **数学训练改变了"平均步长 ↔ 推理正确性" 的关联**. 没经过数学专项调优的模型, 推理时的位移大小与正确性脱钩.

如果未来加 1-2 个其他 math-tuned 小模型, 这个"specific to math-tuning"的 phenomenon 可以独立成立.

### Finding 2: SFT 与 RL-distilled 模型在 `curvature_mean` 上**方向相反**

- **两个 Qwen (SFT-only)**: 正确轨迹更弯 (sign-flipped)
- **DeepSeek (R1-RL-distilled)**: 正确轨迹更直 (not flipped)

→ 假设: **RL 微调把"模型在正确路径上更果断" 这一行为打入了几何 layer**. SFT 模型答对时频繁微调路径 (高曲率); R1-style RL 模型答对时一路推进 (低曲率, 类似它的训练 reward 偏好长且连贯的 CoT).

这一发现, 如果跨更多 SFT/RL 对照模型对验证, 是 paper-worthy phenomenon. **比"找通用错误检测指标"题目更有趣**, 也避开了我们一开始的 baseline 困境.

## 6. Reframing: 从"错误检测" 到"训练范式 → 几何 signature"

原 research_plan.md 把 γ 方向写成 "找出能预测 LLM 推理失败的几何指标". Phase 2 三个模型的数据说: **这个 framing 太朴素了**. 因为:

- 不同模型的"失败"在内部的几何投射方式根本不一样
- 与其找一个通用预测器, 不如**主动操纵训练范式, 观察几何 signature 怎么变**

新 framing 主张:
> **训练范式 (base / SFT / RL / distilled) 在隐状态轨迹的几何上留下了可识别且系统性的指纹. 这些指纹不仅可用于诊断 (区分 "是哪种训练训出来的"), 还能解释为何不同范式在不同任务上表现差异.**

这条线对应的论文模板是 **phenomenon paper**, 类似 induction heads 那种结构:
- 命名一个 phenomenon (e.g. "training-paradigm geometric signatures" 或 "RL-induced trajectory straightening")
- 在多组 model pairs (同 base 不同 post-training) 上严谨刻画
- 提出一个机制性解释 (为什么 RL 让轨迹更直)
- (可选) 设计一个利用此 signature 的下游应用 (例如 model attribution / "这模型是不是 R1 蒸过的?")

## 7. 工程量与下一阶段

### 我们已经具备的能力
- 7 个几何指标, 单机可在分钟级跑 100 题任意 ≤7B 模型的 metrics
- pipeline 完全 config-driven, 加一个新模型 = 写一个 yaml + 跑一条命令
- partial_auc, cross-model aggregation 都有了

### Phase 3 候选 (按 Reframing 调整后)

**方向 B (推荐 — 我的判断)**: 受控对比训练范式对几何的影响
- 准备 4-6 对 (base / SFT / RL) 三元组模型对 (e.g. Llama-3.2-1B vs Llama-3.2-1B-Instruct vs DeepSeek-Distill, 或 Qwen2.5-1.5B-Base vs Instruct vs Math vs R1-Distill)
- 同 100 题 GSM8K + 100 题 non-math (MMLU 子集等) 做交叉
- 主结果: 一个表 (n_models × n_signatures) 显示哪些几何属性在跨模型上 paradigm-cluster

**方向 A (原 plan)**: 继续上 intrinsic dim, OT, PH, 试图找一个通用预测器
- 已经看到迹象表明朴素 framing 行不通, 继续投入风险大

**方向 C (深挖)**: 为什么 `mean_step_norm` 在 math-tuned 模型上独有强信号
- mech-interp 风格, 找具体哪些 token 贡献了 step_norm 差异
- 可能发现: 算术 token 对应的 hidden state 在 math-tuned 模型里有特殊几何

## 8. 失败与教训

### Phase 2 期间唯一新 lesson
- [`2026-05-17_modelscope-per-file-throttling.md`](../../docs/lessons_learned/2026-05-17_modelscope-per-file-throttling.md) — Llama-3.2-1B 因 ModelScope 节点限速 660 kB/s 被中止, 50 分钟下载 74%. 后续选 mirror 必须先做速度 test.

### 复用 fail-fast 模式的收益
- 修了 1 行 (`trust_remote_code`, OSS 直拉, 新 metric 命名) 都重跑只要 ~10 秒, 不是 ~10 分钟
- 一次 extraction 复用了 4 次分析 (Phase 0 验证, Phase 1, Phase 2 preview-2models, Phase 2 cross-model-3)

## 9. 决策记录

**不切方向 γ. 切 γ 内的 framing.**

理由:
1. P1+P2 字面失败, 但 data 给了清晰的 reframing 信号 —— "训练范式 → 几何 signature" 比原 framing 更适合做 phenomenon paper
2. 我们已经有强能力栈 (7 指标 + partial_auc + cross-model aggregation), 直接复用就能做 reframing 后的实验
3. Llama 暂时缺位但不阻塞, 后续补 (或换 mirror)

下一里程碑提案: 方向 B 的"训练范式 controlled comparison". 等用户拍板.

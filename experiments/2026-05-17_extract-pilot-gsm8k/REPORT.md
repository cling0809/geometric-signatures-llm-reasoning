# Phase 0 退出实验: GSM8K hidden-state 提取 pipeline 跑通

- **Exp ID**: 2026-05-17_extract-pilot-gsm8k
- **日期**: 2026-05-17
- **状态**: ✅ 完成
- **配置**: [`configs/2026-05-17_extract-pilot-gsm8k.yaml`](../../configs/2026-05-17_extract-pilot-gsm8k.yaml)
- **Raw outputs**: `~/AI/runs/2026-05-17_extract-pilot-gsm8k/` (项目外, 709 MiB)

## 1. 目的

Phase 0 退出条件: 验证 end-to-end 提取 pipeline 能跑通, 产出符合 schema 的 hidden-state 张量与正确性标签. 给 Phase 1 (几何指标 vs 正确率) 准备数据.

## 2. 设定

| 项 | 值 |
|----|----|
| 模型 | `Qwen/Qwen2.5-Math-1.5B-Instruct` (ModelScope, bfloat16) |
| 数据集 | GSM8K test, 前 20 题 (直接 OSS 拉取) |
| Prompt 模板 | `"Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem: {q}\n\nSolution:"` |
| 解码 | greedy, max_new_tokens=512 |
| 提取 | 全部 29 层 (28 transformer + 1 embedding), 每个生成 token, 取 last-position 隐状态 |
| 硬件 | 1× RTX 5090 (sm_120, 32 GiB), torch 2.12.0+cu130 |

## 3. 关键结果

| 指标 | 值 |
|------|----|
| 样本数 | 20 |
| 准确率 | **65% (13/20)** |
| 平均生成 token 数 | 421 |
| 命中 max_new_tokens=512 上限的样本 | 9 / 20 (45%, 多数对应失败) |
| 平均提取耗时 | 6.0 s/样本 |
| 总运行时间 | ~6 分钟 (含模型下载 3 分钟) |
| 单样本 hidden_states 大小 | ~21–35 MiB (随 n_tokens, bf16) |
| 总 trajectories 大小 | 709 MiB |

**张量 schema 验证** (`sample_0000.pt`):
- `hidden_states.shape == (405, 29, 1536)`, dtype=`torch.bfloat16`
- `prompt_len = 84`, `n_gen_tokens = 405`
- `generated_text` 头部能 round-trip (问题被复述, 然后开始解题)

**labels.parquet schema** (20 行):
- `sample_id, gold, pred, correct, n_gen_tokens`
- pandas / pyarrow 可直接读取

## 4. 结论

✅ Phase 0 退出条件全部达成:
- 20 个 .pt 文件存在且 `load_trajectory()` 全部可重 load
- labels.parquet 读取正常
- DONE 哨兵写入 ("accuracy=0.6500\nn=20")
- pytest 11 passing

✅ **数据集质量适合 Phase 1**:
- 正确/错误样本数 13:7, 类不平衡不严重, AUC 计算有足够 positive/negative
- 两类样本的生成长度分布有差异 (失败样本更多打到 512 上限) —— 这本身可能是一个 trivial baseline, Phase 1 几何指标必须显著超过单纯的"答案长度"才有意义

## 5. 观察到的副产物 (非 Phase 0 范围, Phase 1 处理)

1. **Sample 6**: `pred=inf` —— 文本里有形如 `1e500` 的数被 `float()` 解析成 inf. 需要 `parse_predicted` 增加 `math.isfinite()` 过滤. (优先级低, Phase 1 顺手修)
2. **Sample 15**: 模型走偏, 最后吐了步骤编号 `8096`, 被错认为答案. parse_predicted 的"取最后一个数字"是粗糙的, 但替代方案 (强制 `\\boxed{}`) 在 base 模型上召回率低. 暂不优化, 记到这里
3. **Max-token 截断率 45%**: 推理在 max_new_tokens=512 下经常被截断. Phase 1 是否要拉到 768/1024 需要再权衡 (拉大 = 内存翻倍 + 跑得慢, 但失败样本会更"自然" 而不是被截断)

## 6. 下一步 → Phase 1

按 research_plan.md Phase 1:
- 实现两个最简指标: `trajectory_length`, `mean_curvature` (放在 `src/geoprobe/metrics/`)
- 在当前的 20 题数据上先跑通指标计算 + 算 AUC 框架
- 然后扩到 100 题再跑指标
- 退出条件: AUC > 0.55 → 进入 Phase 2; < 0.55 → 调整指标 / 重新设计提取协议

## 7. 失败重启次数 = 3 (本次 Phase 0 退出阶段)

每次都 fail-fast (< 5 s 内崩), 已记入 lessons_learned:
1. [`2026-05-17_modelscope-msdataset-hidden-deps.md`](../../docs/lessons_learned/2026-05-17_modelscope-msdataset-hidden-deps.md) — addict 等隐式依赖
2. (内联注释) `MsDataset.load` 需要 `trust_remote_code=True`
3. [`2026-05-17_modelscope-msdataset-version-skew.md`](../../docs/lessons_learned/2026-05-17_modelscope-msdataset-version-skew.md) — datasets 库版本与 modelscope shim 不兼容 → 直接走 OSS

这个比例 (3 次工程问题 / 1 次正确运行) 在引入新生态时是正常的. 全部已固化到 `pyproject.toml` / `setup_env.sh` / 代码注释, 下次重装环境不会再踩.

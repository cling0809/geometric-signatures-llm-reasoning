# <实验标题>

- **Exp ID**: YYYY-MM-DD_<slug>
- **日期**: YYYY-MM-DD
- **状态**: 进行中 / 完成 / 失败 / 废弃
- **作者**:
- **配置**: `configs/<file>.yaml` 或下方内联
- **Raw outputs**: `~/AI/runs/<exp-id>/`

## 1. 目的

1-2 句话: 这个实验回答什么问题. 对应 research_plan.md 哪个假设 / 里程碑.

## 2. 设定

- 模型:
- 数据集 / 样本数:
- 提取协议 (哪些层, 哪些 token):
- 指标:
- 软硬件: (GPU, torch 版本, commit hash)

## 3. 关键结果

- 数字 (表):

| 模型 | 指标 X | 指标 Y |
|------|--------|--------|
|      |        |        |

- 关键图 (≤3 张, 放 `plots/`):

## 4. 结论

- 假设 H_i 是否得到支持?
- 与预期偏差:
- 是否进入下一 phase / 是否触发 fallback:

## 5. 下一步

- [ ] ...

## 6. (如失败) 失败分析

- 失败模式:
- 根因:
- 教训 (已写入 `docs/lessons_learned/YYYY-MM-DD_<title>.md`):

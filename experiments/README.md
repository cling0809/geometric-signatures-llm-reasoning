# experiments/

每个实验在此目录下有一个**子目录** `<exp-id>/`, 存放**实验的描述与结论** (不存大数据).

实验 ID 格式: `YYYY-MM-DD_<短语义 slug>`, 例: `2026-05-17_curvature-pilot-gsm8k`.

## 每个 `<exp-id>/` 必须包含

```
experiments/<exp-id>/
├── REPORT.md          # 实验报告 (见 ../TEMPLATE.md)
├── config.yaml        # 实验配置 (或链接到 configs/<exp-id>.yaml)
└── plots/             # 关键图 (≤几张, 大图放 runs/)
```

**注意区分**:
- `experiments/<exp-id>/` 入 git, 长期保留 —— 描述 + 结论
- `~/AI/runs/<exp-id>/` 不入 git, 可丢弃 —— 大文件 (hidden states, logs, parquet 等)

## 何时新建实验

任何一次"有明确目的的运行" —— 哪怕只跑 10 个样本验证 pipeline —— 都应建实验目录. 不建的运行就是不存在的运行.

## 何时删除实验

按 WORKING_RULES §4:
- 如果实验失败但有教训: 删 `runs/<exp-id>/` 的大文件, **保留** `experiments/<exp-id>/REPORT.md`, 同时在 `docs/lessons_learned/` 加索引
- 如果实验完全无价值 (例如错误命令导致没产出): 删整个 `experiments/<exp-id>/`

# SmolVLA 项目目录规范

## 项目根目录

所有项目任务统一从以下目录执行：

```text
/path/to/we_smolvla
```

该目录同时包含 LeRobot 源码和 SmolVLA 项目资产。LeRobot Python 包源码位于：

```text
/path/to/we_smolvla/src/lerobot
```

## 固定目录

```text
/path/to/we_smolvla/
├── docs/
│   ├── 00_总方案计划.md
│   ├── 01_部署项目指导书.md
│   ├── 02_仿真测试项目指导书.md
│   ├── 03_数据收集与格式处理项目指导书.md
│   ├── 04_微调项目指导书.md
│   ├── 05_本地微调结果测试指导书.md
│   ├── PROJECT_LAYOUT.md
│   └── source/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── models/
│   ├── smolvla_base/
│   └── smolvla_libero/
├── outputs/
│   ├── train/
│   └── eval/
├── logs/
├── reports/
└── scripts/
```

## 路径职责

- `docs/`：项目方案、操作指导书和上游 LeRobot 文档。
- `datasets/raw/`：尚未整理的原始数据。
- `datasets/processed/`：字段规范化、清洗、拆分后的数据集。
- `datasets/eval/`：策略真机评测数据，不参与训练。
- `models/`：基础模型和可独立复用的预训练模型。
- `outputs/train/`：训练检查点、优化器状态和训练配置。
- `outputs/eval/`：仿真评估结果、视频和指标文件。
- `logs/`：终端训练日志和后台任务日志。
- `reports/`：部署记录、测试表、失败分析和汇总报告。
- `scripts/`：可复用的数据处理、评估和统计脚本。

Hugging Face 和 LeRobot 的系统缓存继续使用：

```text
$HOME/.cache/huggingface
$HOME/.cache/huggingface/lerobot
```

该缓存不属于项目产物，但包含已经下载的模型、校准文件和运行时资产。除非确认需要
迁移并备份，否则不要删除或清空。

## 命令约定

进入项目根目录：

```bash
cd /path/to/we_smolvla
conda activate lerobot_043
```

数据集使用绝对路径，原始副本放在 `raw/`，清洗和字段规范化后的数据放在
`processed/`：

```bash
--dataset.repo_id=local/<dataset_name>
--dataset.root=/path/to/we_smolvla/datasets/raw/<dataset_name>
# 或
--dataset.root=/path/to/we_smolvla/datasets/processed/<dataset_name>
```

训练输出使用绝对路径：

```bash
--output_dir=/path/to/we_smolvla/outputs/train/<job_name>
```

评测数据使用独立名称和独立目录：

```bash
--dataset.repo_id=local/eval_<task>_<model>_<step>
--dataset.root=/path/to/we_smolvla/datasets/eval/eval_<task>_<model>_<step>
```

训练日志建议同时保存到文件：

```bash
lerobot-train \
  ... \
  2>&1 | tee /path/to/we_smolvla/logs/<job_name>.log
```

## 运行约束

- 不在 `smolvla` 目录创建新的数据、模型、输出或日志。
- 不在训练、录制或评估进程运行时移动对应数据目录。
- 每个检查点使用独立的评测数据名称和目录。
- 文档中的绝对路径必须与磁盘实际路径一致。
- `repo_id` 表示数据集逻辑名称，`root` 表示数据集物理目录，两者必须成对使用。

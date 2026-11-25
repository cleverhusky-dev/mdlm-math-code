# MDLM 数学代码监督微调

输入数学题，由 MDLM 生成 Python 程序。验证时执行程序并比较数值答案，按正确率选择最佳检查点。这里保留实验 0–12 的配置和运行入口。实验 0 评估预训练模型，实验 12 使用统一变量名的多解训练数据。

## 环境与数据

正式训练使用 Linux、Python 3.12、PyTorch 2.6.0+cu124 和两张 RTX 4090。训练采用双卡 DDP，正式生成评测使用单卡。以下命令均在本目录执行。

```bash
bash setup.sh --download
source .venv/bin/activate
python scripts/rebuild_data.py vars --download
python scripts/manage_inputs.py
```

安装脚本按 [输入锁](inputs.lock.json) 获取 s-flm 预训练权重和 SmolLM 分词器。数据脚本按 [数据来源锁](datasets/sources.lock.json) 获取固定版本的公开数据，在 `data/` 下写入训练及验证文件。已缓存的输入先核对 SHA-256，再复用。仅准备数据或运行 CPU 测试时，可以在已安装 `requirements.txt` 的 Python 3.12 环境中先执行 `python scripts/manage_inputs.py --upstream-only --skip-checkpoint --download` 获取分词器。

| 数据版本 | 对应实验 | 训练代码样本 |
| --- | --- | ---: |
| `prepared` | 0–5，GSM8K-PoT 清洗后按题目划分 | 12,953 |
| `style` | 6–9，统一代码风格并去除同题重复代码 | 12,288 |
| `mix` | 10，在 style 基础上混入 20% TinyGSM 样本 | 15,360 |
| `single` | 11，每道训练题只选一份解法 | 5,747 |
| `vars` | 12，将中间变量依次命名为 `x_1`、`x_2` 等 | 12,288 |

`scripts/rebuild_data.py` 接受上表中的版本名并自动构建依赖。各版本共用 639 道验证题。若需全部实验数据，运行 `python scripts/rebuild_data.py all --download`。其中只有混合数据实验需要下载约 3.75 GB 的 TinyGSM 原始分片。官方 GSM8K 测试文件也可单独用 `python scripts/rebuild_data.py test --download` 获取。

清洗索引记录通过历史执行核验的 MathInstruct 来源行，脚本从上游恢复题目和代码，再次核对官方训练题及标准答案。混合数据重放固定随机种子选出的候选，并按代码哈希关联留存的 Linux 执行记录。完整审计与执行工具分别见 `scripts/cleaning/` 和 `scripts/prepare_mix_data.py`。每个数据版本均核对两个分区的 8 个历史文件哈希、样本数和编码指纹，再写入跨目录稳定的发布清单。

## 训练与评测

```bash
bash experiments/experiment12/run.sh --dry-run
bash experiments/experiment0/run.sh
bash experiments/experiment12/run.sh

python scripts/rebuild_data.py test --download
python scripts/freeze_benchmark.py --sft-dir runs/experiment12 --output work/benchmark.json
bash benchmarks/gsm8k/run.sh --config work/benchmark.json
```

实验 12 从预训练 EMA 权重初始化，使用普通模型权重训练与评估。它训练 5 轮，峰值学习率为 `1e-4`，有效全局批次为 64。正式训练依赖同一版代码完成的实验 0 基座评测。实验 3 和 5 使用实验 1 训练满三轮的第 609 步权重，实验 4 的启动脚本运行高、低学习率两组。输出写入 `runs/`。模型选择和原始 GSM8K 测试成绩见 [实验记录](../docs/experiments.md)。

官方测试共 1,319 题，固定比较预训练模型与验证集选出的 SFT 模型，使用三组温度和去噪步数。冻结命令将本次选出的步数、模型哈希和文件哈希写入新的评测配置。RL 使用同一个实验 12 的最佳权重，见 [强化学习](../rl/README.md)。

## 代码入口

| 路径 | 内容 |
| --- | --- |
| `mdlm_sft/` | 数据编码、DiT 模型、去噪目标、采样、训练、执行评分及检查点 |
| `scripts/`、`datasets/` | 固定数据来源、重建和完整性核验 |
| `experiments/`、`benchmarks/gsm8k/` | 实验配置、启动脚本及官方测试协议 |
| `tests/` | 数据变换、训练状态、采样、评分和运行入口的回归测试 |

准备好分词器后，运行 `python -B -m unittest discover -s tests -v`。模型和采样实现沿用 [s-flm](https://github.com/jdeschena/s-flm/tree/30def79976ca3b1cdcb5fcbfa0165524e44001de) 的 MDLM 配置，模型输入与数据版本见两份锁文件。许可与来源见 [项目说明](../licenses/)。

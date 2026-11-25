# 用 diffu-GRPO 微调 MDLM

从 SFT 实验 12 的最佳模型出发，用程序执行结果训练模型解数学题。原实验共完成 1,000 次策略更新，每 100 步在固定的 639 道验证题上评估。使用相同的生成设置时，初始 SFT 模型答对 290 题（45.38%），第 500 步答对 308 题（48.20%），第 1,000 步答对 294 题（46.01%）。逐步结果见 [RL 实验结果](../results/rl/)。

## 方法与配置

[实验配置](experiments/experiment1/config.json) 固定题目划分、生成协议和训练参数。[目标函数](mdlm_rl/objective.py) 与[训练循环](mdlm_rl/train.py) 实现了以下步骤。

1. 每个 rollout 取 8 道训练题，每题采样 8 份代码。题目保持可见，模型以温度 0.1 进行 128 步祖先去噪采样。
2. 在 Linux 隔离执行器里运行代码并核对数值。答对奖励为 1，其他情况为 0。每份程序的优势等于本次奖励减去同题 8 份程序的平均奖励。
3. 在估计已生成 token 的对数概率时，遮盖全部代码，并以 0.15 的概率遮盖题目 token。当前策略、旧策略和参考策略共用同一掩码。每次内部更新使用新的掩码，打分温度为 1。
4. 每个 rollout 执行 4 次策略更新，裁剪系数为 0.2，KL 系数为 0.04。参考策略固定为初始化的 SFT 模型。使用 AdamW，学习率在前 20 步升至 `3e-6`，之后保持恒定。

双卡 BF16 训练每 100 步评估完整验证集，正确题数严格增加时更新最佳检查点。原实验共生成 16,000 份训练代码，双卡耗时约 5 小时。以上分数均来自一次固定随机种子的验证评估。

## 环境与数据

训练和代码评分使用 Linux、Python 3.12、PyTorch 2.6.0、两张支持 BF16 的 GPU。原实验使用 2 × RTX 4090 24 GB。以下命令从仓库根目录进入 `rl/` 后执行。

```bash
cd rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

[data/gsm8k/](data/gsm8k/) 包含从 GSM8K 官方训练集划分出的 5,747 道训练题和 639 道验证题，每题保留题目、题目编号和标准数值答案。[inputs/tokenizer/](inputs/tokenizer/) 包含固定的 SmolLM tokenizer。源数据和分词器的上游授权见[第三方许可](../licenses/)。

如需从 SFT 的 `data/vars/` 重新导出相同题目，可运行 `python scripts/prepare_data.py`。导出文件及其来源指纹默认保存在 `runs/data/gsm8k/`。

## 从 SFT 模型开始训练

先按 [SFT 工程说明](../sft/README.md) 完成实验 12，取得自己的 `best.pt`。同一配置重新训练得到的权重可能有不同的步数与文件哈希。以下命令读取该检查点，核对训练数据身份，并生成本地 RL 配置。使用自定义 SFT 输出目录时，替换 `--checkpoint` 的路径。仅加载自己生成或可信来源的完整 PyTorch 检查点。

```bash
python -B scripts/configure_run.py \
  --checkpoint ../sft/runs/experiment12/training/checkpoints/best.pt \
  --sft-data ../sft/data/vars
config=runs/config.local.json

python -B scripts/run_rl.py --config "$config" --dry-run
bash experiments/experiment1/run.sh --config "$config" --preflight-only
bash experiments/experiment1/run.sh --config "$config" --smoke
bash experiments/experiment1/run.sh --config "$config"
```

`--dry-run` 在 CPU 上检查权重、题目、tokenizer 与配置。预检查还会验证 GPU、分布式环境和隔离执行器。短跑执行 8 次更新，默认写入 `runs/experiment1-smoke/`。正式训练默认写入 `runs/experiment1/`，其中 `checkpoints/best.pt` 保存最佳模型，`checkpoints/last.pt` 保存恢复训练所需的完整状态。

```bash
bash experiments/experiment1/run.sh --config runs/config.local.json \
  --resume runs/experiment1/checkpoints/last.pt
```

初始实验配置记录了原始 SFT 第 960 步权重的 SHA-256。使用该历史检查点时，可将其置于配置指定位置，并直接运行启动器。启动脚本也可通过 `PYTHON_BIN` 指定现有 Python 环境。

## 验证实现

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

测试覆盖组内优势、代理目标、KL、梯度累积、双进程 Gloo 的 token 归一化、检查点恢复和启动逻辑。模型代码基于 [s-flm 的固定版本](https://github.com/jdeschena/s-flm/tree/30def79976ca3b1cdcb5fcbfa0165524e44001de)，策略优化适配了 [d1 的 diffu-GRPO 方法](https://github.com/dllm-reasoning/d1)。

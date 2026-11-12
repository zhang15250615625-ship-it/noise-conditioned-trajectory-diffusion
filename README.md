# Noise-Conditioned Trajectory Diffusion

**噪声条件卷积与局部轨迹约束的扩散图像生成研究。**

本项目使用 PyTorch 实现 CIFAR-10 像素空间扩散模型，研究两个改动：根据当前噪声强度调节 U-Net 的卷积核，以及用冻结教师的局部轨迹辅助训练学生模型。仓库提供整理后的模型代码、训练和采样入口、CPU 测试及已经完成实验的结果摘要。

## 方法

- **基础模型**：32×32 RGB U-Net，64 个训练时间步，v 预测，EMA，评价时使用 31 步 DDIM。
- **噪声条件卷积**：在解码器四个卷积层中加入零初始化修正核，使用 `W(t) = W_base + g(t) W_delta`，其中 `g(t) = 2 sigmoid(-logSNR(t)) - 1`。基础模型约 106 万参数，条件模型约 127 万参数。
- **局部轨迹约束**：学生在局部区间内更新两次，对齐冻结教师更新四次得到的终点，与原 v 预测损失共同训练。支持跨度 4、跨度 8 和共同起点控制。两次局部更新不代表完整图像能够两步生成。

基础 U-Net 来自第三方实现，许可证和来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Haar 变换用于内部张量表示，网络本身仍在像素空间计算。

## 已完成的实验

在锁定模型后的 CIFAR-10 独立测试参考评价中，种子 22 的组合模型相对基线，在 E/F 两组生成噪声下的 KID 分别降低 **8.84% 和 4.80%**。这是选定种子的记录，不是跨训练种子的平均改善。

共同起点对照中，较长局部区间在三个训练种子、两组生成噪声下均改善低频和局部块统计距离，KID 有 4/6 条件改善。组合模型相对仅结构模型没有一致的额外 KID 优势。

具体比较对象、原始指标和选择规则见 [实验说明](docs/EXPERIMENTS.md) 与 [results/](results/)。结果文件来自原实验归档；当前发布包不包含历史检查点、数据集和 Inception 权重。

## 安装

使用 Python 3.10 或更新版本，在解压后的项目目录运行：

```bash
python -m venv .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

然后安装项目：

```bash
python -m pip install -e .
```

CPU 核验环境为 Python 3.10.20、PyTorch 1.13.1、NumPy 1.26.4、Pillow 12.2.0。使用 GPU 时，请先安装与本机驱动匹配的 PyTorch。项目不自动占用、暂停或恢复其他 GPU 任务。

## 无数据 CPU 烟测

```bash
python -m unittest discover -s tests -v
python -m nctd.train --synthetic --steps 2 --batch 2 --device cpu --output runs/smoke
python -m nctd.sample --checkpoint runs/smoke/checkpoint_000002.pt --output samples/smoke --count 4 --steps 4 --device cpu
```

合成数据烟测只验证代码可以运行，生成图不能作为图像质量结果。正式采样默认 31 步。

## 使用 CIFAR-10 训练

从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 Python 版本，解压至 `data/cifar-10-batches-py/`。加载器校验五个训练文件的 MD5，按固定种子划分 45000 张训练图像和 5000 张开发图像。训练不读取测试批次。

基础模型和条件卷积模型分别训练：

```bash
python -m nctd.train --config configs/dense.json --data-root data --device cuda:0 --output runs/dense60k
python -m nctd.train --config configs/noise.json --steps 60000 --data-root data --device cuda:0 --output runs/noise60k
```

从条件模型的 EMA 权重开始，分别运行同样 10000 步的无辅助损失控制和局部轨迹辅助训练：

```bash
python -m nctd.train --config configs/noise.json --steps 10000 --data-root data --init-from runs/noise60k/checkpoint_060000.pt --device cuda:0 --output runs/noise_control10k
python -m nctd.train --config configs/trajectory_span4.json --data-root data --init-from runs/noise60k/checkpoint_060000.pt --device cuda:0 --output runs/trajectory4_10k
```

跨度 8 与共同起点跨度 4 的配置分别为 `configs/trajectory_span8.json`、`configs/trajectory_common_start4.json`。它们都应从同一个父检查点开始。`static.json` 提供固定修正系数的结构控制。基础模型的继续训练控制同样可以使用 `dense.json` 配合 `--init-from` 和 `--steps 10000`。若运行仅轨迹约束的对照，将轨迹配置中的 `variant` 改为 `dense`，并使用对应 dense 父模型。

新训练入口保留核心计算，但不是旧实验调度器的完整复制。以上命令是重新训练方法的入口，不能直接保证重现归档分数。详见 [复现范围](docs/REPRODUCIBILITY.md)。

## 恢复与采样

`--resume` 恢复模型、优化器、EMA、冻结教师和随机状态；`--steps` 表示恢复后累计达到的总步数。输出到新的目录，避免覆盖已有结果。下面从 CPU 烟测的第 2 步继续到第 4 步：

```bash
python -m nctd.train --resume runs/smoke/checkpoint_000002.pt --synthetic --steps 4 --output runs/smoke_resumed --device cpu
```

`--init-from` 只用于从父模型 EMA 开始新的训练阶段，会重新建立优化器与随机状态。它与断点恢复不同。正式的轨迹辅助训练需要已有父模型，程序会拒绝没有父模型的真实数据轨迹训练。

```bash
python -m nctd.sample --checkpoint runs/trajectory4_10k/checkpoint_010000.pt --output samples/trajectory4 --count 64 --steps 31 --device cuda:0
```

训练检查点采用 PyTorch 序列化，仅读取自己生成或可信来源的检查点。

## 评价

`nctd.metrics` 提供低频、局部块切片 Wasserstein 距离和原实验使用的多项式 KID 计算。评价输入为浮点 NCHW `.npy` 图像，真实参考像素缩放至负 1 到正 1：

```bash
python -m nctd.evaluate --generated samples/trajectory4/raw_samples.npy --reference reference.npy --output evaluation.json
```

若另行准备了相同网络、权重、预处理下提取的特征，可同时传入 `--generated-features generated_features.npy --reference-features reference_features.npy` 计算 KID。包内没有 Inception 提取器；不提供特征时不会计算或虚构 KID。历史 KID 的特征配置见 [results/metrics.json](results/metrics.json)。

## 目录

```text
src/nctd/          模型、目标函数、训练、采样和评价
configs/           基础、结构与局部轨迹配置
tests/             不依赖历史权重和数据集的 CPU 测试
results/           脱敏后的真实实验结果
docs/              实验说明、复现范围和来源记录
licenses/          第三方许可证原文
```

GitHub 上传操作见 [上传说明](docs/UPLOAD.md)。本压缩包不包含简历文案。

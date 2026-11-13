# More Denoising Steps Can Hurt: Timestep Selection in Diffusion-Based OTFS Channel Estimation

English: [README.md](README.md)

扩散模型做MIMO-OTFS信道估计时，去噪步数并不是越多越好。论文比较前置（leading）与尾置（trailing）时间步间隔，说明一步尾置DDIM就比默认五步更准，并给出按导频信噪比匹配起始时间步的规则，在未调参的信噪比上也能预测最优起点。


## 目录

| 目录 | 内容 |
|---|---|
| `code/` | 仿真与模型代码（`otfs5/`是公共库，其余是本文的实验脚本） |
| `paths/` | DeepMIMO v3射线追踪路径（9个城市，纽约作为未见城市；使用条款见`paths/README.md`） |
| `configs/runs/` | 本文每个网络的训练配置（原始训练时保存的config.json） |
| `pipeline/` | 一键运行的步骤：`train_run.py`训练，`steps.sh`评估，`check_results.py`对比 |
| `reference_results/` | 论文里数字的出处（原始结果），用于对比你复现的结果 |
| `results/v21/`，`runs_v21/`，`lmmse_v21/` | 运行后生成：结果、网络、LMMSE锚点 |

## 环境

需要Linux和NVIDIA GPU（论文结果在RTX PRO 6000 96 GB上得到）。

```
conda create -n otfs python=3.10 -y && conda activate otfs
# 先按 https://pytorch.org 安装与你的CUDA匹配的torch 2.9.1，再：
pip install -r requirements.txt
```

## 一键运行

```
bash run_all.sh smoke    # 先做环境检查：GPU上几分钟；没有GPU时加 OTFS_DEV=cpu，约半小时
bash run_all.sh weights  # 下载论文训练好的网络（233 MB，GitHub release），之后 eval、check 不必训练
bash run_all.sh          # 完整复现：训练全部网络，跑全部评估，最后和论文数字对比
```

也可以分步：`bash run_all.sh train`、`bash run_all.sh eval`、`bash run_all.sh check`。用哪块GPU由`CUDA_VISIBLE_DEVICES`决定，python由`PY`指定，例如：

```
CUDA_VISIBLE_DEVICES=0 PY=~/miniconda3/envs/otfs/bin/python bash run_all.sh
```

中断后重新运行同一条命令即可：训练好的网络（目录里有`DONE`）和已有的结果会自动跳过，训练到一半的网络从`last.pt`继续。某一步评估失败不会中断其余步骤：`eval`会报告它、继续跑完剩下的步骤，并在最后列出全部失败项（日志在`results/v21/`）。长任务请放在`tmux`或`screen`里跑。

## 需要的算力

本文共训练6个网络，按每步约0.5秒估计，单卡训练约38 GPU小时，另加锚点构建和评估若干小时；用`run_all.sh weights`可免去训练，只需评估，但锚点构建和全部评估仍要若干GPU小时、占数十GB硬盘。LMMSE锚点按信噪比网格缓存在`lmmse_v21/`，可能占用数十GB硬盘，构建联合锚点时显存建议不少于48 GB。

## 结果与论文的对应

| 论文内容 | 结果文件（`results/v21/`，参考值在`reference_results/`） |
|---|---|
| 图1与步数研究 | p1_steps.json，main_std.json |
| 表I（首个时间步消融，本文模型） | p1_ablation.json，p1_ablation_eps_last.json |
| 表I（已发表CRDM模型）与1000帧复核 | v1_ablation.json，v1check_spacing_n1000.json（需要原始CRDM数据，见“注意”） |
| 表II（信噪比匹配起点） | p1_snrstart_v.json，p1_snrstart_eps.json，p1_snrstart_eps_ext.json，p1_r3_eps.json，p1_r3_v.json |
| 未调参信噪比上的起点预测 | p1_r4_eps.json（CPU） |
| 单流误码率 | p1_ber_ss.json |

`bash run_all.sh check`会逐个文件比较你的结果和参考值，列出最大差值。重新训练的网络和不同的GPU会带来小差异，一般在百分之几到十分之一dB，论文的结论不应改变。

## 注意

- 每个结果对应的命令都在`pipeline/steps.sh`里，按论文顺序排列，可以单独复制出来运行。
- 带`(CPU)`的步骤按论文的做法在CPU上运行（`OTFS_DEV=cpu`）。CPU和GPU的随机数生成器不同，抽到的测试帧也不同，所以这些步骤要和CPU的参考值比。
- 表I中已发表CRDM模型的两行和1000帧复核，需要CRDM原始数据集（`multiscene_otfs_v4/test_seen`）和已发表的检查点。运行前设置`V1_DATA=<数据目录>`和`V1_CKPT=<best_model.pth>`，否则这两步会自动跳过。
- 路径数据来自DeepMIMO v3数据集（ITA 2019），使用时请引用。

## 许可

代码：MIT（见`LICENSE`）。射线追踪路径：见`paths/README.md`。

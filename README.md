# More Denoising Steps Can Hurt: Timestep Selection in Diffusion-Based OTFS Channel Estimation

Code for the paper of the same title (under review). The preprint link will be added here.

For diffusion-based MIMO-OTFS channel estimation, more denoising steps are not always better. The paper compares leading and trailing timestep spacing, shows that one trailing DDIM step is more accurate than the default five, and gives a rule that matches the starting timestep to the pilot SNR, which also predicts the best start at SNRs that were not tuned.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Simulator and models (`otfs5/` is the shared library, the other scripts are the experiments of this paper) |
| `paths/` | DeepMIMO v3 ray-tracing paths, nine cities, New York held out as the unseen city (see `paths/README.md` for terms) |
| `configs/runs/` | Training configuration of every network of the paper (the `config.json` saved at training time) |
| `pipeline/` | The one-command steps: `train_run.py` trains, `steps.sh` evaluates, `check_results.py` compares |
| `reference_results/` | The raw results behind the numbers in the paper, to compare your run against |
| `results/v21/`, `runs_v21/`, `lmmse_v21/` | Created by a run: results, networks, LMMSE anchors |

## Environment

Linux with an NVIDIA GPU. The results of the paper were obtained on an RTX PRO 6000 with 96 GB.

```
conda create -n otfs python=3.10 -y && conda activate otfs
# install torch 2.9.1 for your CUDA version from https://pytorch.org, then:
pip install -r requirements.txt
```

## Reproduce

```
bash run_all.sh smoke                     # environment check: minutes on a GPU, about half an hour with OTFS_DEV=cpu
bash run_all.sh weights                   # download the trained networks of the paper (200 MB release asset)
bash run_all.sh eval && bash run_all.sh check   # rerun every evaluation and compare with the paper
bash run_all.sh                           # or the full reproduction: train every network, evaluate, compare
```

The stages can also be run one at a time: `train`, `eval`, `check`. `CUDA_VISIBLE_DEVICES` picks the GPU and `PY` the python, for example

```
CUDA_VISIBLE_DEVICES=0 PY=~/miniconda3/envs/otfs/bin/python bash run_all.sh
```

Rerunning the same command after an interruption continues where it stopped: finished networks (a `DONE` file in their folder) and existing results are skipped, and a network that was half trained resumes from `last.pt`. Run long jobs inside `tmux` or `screen`.

## Compute

Training all 6 networks of the paper takes about 38 GPU hours on one GPU (about 0.5 s per step), plus a few hours for anchors and evaluation. With `run_all.sh weights` the training is skipped and only the evaluation remains: building the anchors and rerunning every evaluation still takes a few GPU hours and tens of GB of disk. The LMMSE anchors are cached per SNR in `lmmse_v21/` and can take tens of GB of disk; building the joint anchor needs at least 48 GB of GPU memory.

## Results and the paper

Table numbers refer to the submitted version.

| In the paper | Result file (`results/v21/`, reference in `reference_results/`) |
|---|---|
| Fig. 1 and the step study | p1_steps.json, main_std.json |
| Table I, first-timestep ablation (this model) | p1_ablation.json, p1_ablation_eps_last.json |
| Table I, published CRDM model, and the 1000-frame recheck | v1_ablation.json, v1check_spacing_n1000.json (needs the original CRDM data, see Notes) |
| Table II, SNR-matched start | p1_snrstart_v.json, p1_snrstart_eps.json, p1_snrstart_eps_ext.json, p1_r3_eps.json, p1_r3_v.json |
| Start prediction at untuned SNRs (CPU) | p1_r4_eps.json |
| Single-stream bit error rate | p1_ber_ss.json |

`bash run_all.sh check` compares every result file with its reference and lists the largest difference. Retrained networks and a different GPU give small differences, typically a few hundredths to a tenth of a dB; the conclusions of the paper should not change.

## Notes

- Every command behind a result is in `pipeline/steps.sh`, in the order of the paper, and can be copied out and run on its own.
- Steps marked `(CPU)` run on the CPU as in the paper (`OTFS_DEV=cpu`). CPU and GPU random generators draw different test frames, so compare those steps with the CPU reference values.
- The two rows of Table I that use the published CRDM model, and the 1000-frame recheck, need the original CRDM test set (`multiscene_otfs_v4/test_seen`) and its checkpoint, which are not part of this repository. Set `V1_DATA=<data folder>` and `V1_CKPT=<best_model.pth>` to run them; otherwise these two steps are skipped.

## License

Code: MIT (see `LICENSE`). Ray-tracing paths: see `paths/README.md`.

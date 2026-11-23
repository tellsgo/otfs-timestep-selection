# Timestep Selection in Diffusion-Based OTFS Channel Estimation

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
| `results/v22/`, `runs_v22/`, `lmmse_v22/` | Created by a run: results, networks, LMMSE anchors |

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
bash run_all.sh weights                   # download the trained networks of the paper (233 MB release asset)
bash run_all.sh eval && bash run_all.sh check   # rerun every evaluation and compare with the paper
bash run_all.sh                           # or the full reproduction: train every network, evaluate, compare
```

The stages can also be run one at a time: `train`, `eval`, `check`. `CUDA_VISIBLE_DEVICES` picks the GPU and `PY` the python, for example

```
CUDA_VISIBLE_DEVICES=0 PY=~/miniconda3/envs/otfs/bin/python bash run_all.sh
```

Rerunning the same command after an interruption continues where it stopped: finished networks (a `DONE` file in their folder) and existing results are skipped, and a network that was half trained resumes from `last.pt`. A failed evaluation step does not stop the others: `eval` reports it, runs the remaining steps and lists every failure at the end (its log is in `results/v22/`). Run long jobs inside `tmux` or `screen`.

## Compute

Training all 6 networks of the paper takes about 38 GPU hours on one GPU (about 0.5 s per step), plus a few hours for anchors and evaluation. With `run_all.sh weights` the training is skipped and only the evaluation remains: building the anchors and rerunning every evaluation still takes a few GPU hours and tens of GB of disk. The LMMSE anchors are cached per SNR in `lmmse_v22/` and can take tens of GB of disk; building the joint anchor needs at least 48 GB of GPU memory.

## Results and the paper

The rows follow the long (arXiv) version. The letter version contains a subset of the same results.

| In the paper | Result file (`results/v22/`, reference in `reference_results/`) |
|---|---|
| Step count: NMSE versus the number of steps (figure) and the latency table | p1_steps.json, main_std.json |
| Contribution of the first timestep: spacing table and bootstrap intervals (this model) | p1_ablation.json, p1_ablation_eps_last.json |
| Spacing table, rows of the published CRDM network | v1_ablation.json (needs the original CRDM data, see Notes) |
| Anchor-error-matched start and the noise-prediction network | p1_snrstart_v.json, p1_snrstart_eps.json, p1_snrstart_eps_ext.json, p1_r3_eps.json, p1_r3_v.json |
| Start prediction at untuned SNRs (CPU) | p1_r4_eps.json |
| Bit error rate: per-bin rank of the 4x4 channel | rank_stats.json |
| Bit error rate: single-stream configuration | p1_ber_ss.json |
| The published CRDM network (1000 frames, and the first 250-frame run) | v1check_spacing_n1000.json, v1check_spacing_n250.json (needs the original CRDM data) |

`bash run_all.sh check` compares every result file with its reference and lists the largest difference. Retrained networks and a different GPU give small differences, typically a few hundredths to a tenth of a dB; the conclusions of the paper should not change.

## Notes

- Every command behind a result is in `pipeline/steps.sh`, in the order of the paper, and can be copied out and run on its own.
- Steps marked `(CPU)` run on the CPU as in the paper (`OTFS_DEV=cpu`). CPU and GPU random generators draw different test frames, so compare those steps with the CPU reference values.
- The results of the published CRDM network need the original CRDM test set (`multiscene_otfs_v4/test_seen`) and its checkpoint, which are not part of this repository. Set `V1_DATA=<data folder>` and `V1_CKPT=<best_model.pth>` to run them; otherwise these steps are skipped. Those results come from the published network and its own data, so they did not change in v22.
- v22 (this version, release v1.2): the path power of the ray-traced paths is linear (watts), and `sim.py` now takes its square root as the amplitude. Earlier versions treated it as dB, which gave every path nearly the same amplitude. All networks were retrained and every evaluation was rerun; `configs/runs/` holds the new training records.

## License

Code: MIT (see `LICENSE`). Ray-tracing paths: see `paths/README.md`.

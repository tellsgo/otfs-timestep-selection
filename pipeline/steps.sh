#!/bin/bash
# Evaluations behind the paper, in order.  A result that already exists is skipped (delete it to recompute).
# Run through run_all.sh (it sets ROOT and PY).  CPU-only steps use OTFS_THREADS threads (default 16).
# A failed step is reported and the next steps still run; the summary at the end lists the failures (exit 1 if any).
cd "$ROOT/code" || exit 1; R="$ROOT/runs_v22"; O="$ROOT/results/v22"
ev() { local out="$O/$1"; shift; if [ -f "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
       echo "[eval] $(basename "$out")"; "$@" > "$out.log" 2>&1 && { DONE_N=$((DONE_N+1)); return; }
       echo "[FAILED] $(basename "$out"), see $out.log"; FAILED+=("$(basename "$out")"); }
DONE_N=0; FAILED=()
summary() { echo "== $DONE_N steps computed now, ${#FAILED[@]} failed${FAILED:+: ${FAILED[*]}}"; [ ${#FAILED[@]} -eq 0 ]; }
CPU="env CUDA_VISIBLE_DEVICES= OTFS_DEV=cpu OTFS_THREADS=${OTFS_THREADS:-16}"
EV="$PY -m otfs5.evaluate"
# Fig. 1 and the step study; main grid of the anchored diffusion model and the regression network
ev main_std.json $EV --out $O/main_std.json --methods ls,thr,gsomp,lmmse --nets crdmS=$R/diffS_std/best.pt,crdmA=$R/diffA_std/best.pt,regA=$R/regA_std/best.pt --steps 1t,5 --n 1024 --snrs 0,5,10,15,20 --speeds 60,120,250,375,500,650,800 --ber --splits test,unseen
ev p1_steps.json $EV --out $O/p1_steps.json --methods lmmse --nets crdmA=$R/diffA_std/best.pt,regA=$R/regA_std/best.pt --steps 1,2,3,5,10,20,50,1t,2t,3t,5t,10t,20t,50t,1t:4,1t:8,5t:8 --n 512 --snrs 0,5,10,15,20 --speeds 500 --splits test,unseen
# Table I (first-timestep ablation) for the v model and for the epsilon model (final EMA weights with their config)
ev p1_ablation.json $PY p1_ablation.py
[ -f $R/diffE_std/last_cfg.pt ] || $PY -c "
import torch
b = torch.load('$R/diffE_std/best.pt', map_location='cpu', weights_only=False)
l = torch.load('$R/diffE_std/last.pt', map_location='cpu', weights_only=False)
torch.save(dict(ema=l['ema'], scale=l['scale'], step=l['step'], val=None, config=b['config']), '$R/diffE_std/last_cfg.pt')" \
  || { echo "[FAILED] diffE_std/last_cfg.pt (needs runs_v22/diffE_std/last.pt)"; FAILED+=(diffE_std/last_cfg.pt); }
ev p1_ablation_eps_last.json $PY p1_ablation.py $R/diffE_std/last_cfg.pt $O/p1_ablation_eps_last.json
# Table II: SNR-matched start
ev p1_snrstart_v.json $PY p1_snrstart.py $R/diffA_std/best.pt $O/p1_snrstart_v.json
ev p1_snrstart_eps.json $PY p1_snrstart.py $R/diffE_std/last_cfg.pt $O/p1_snrstart_eps.json
ev p1_snrstart_eps_ext.json env KAPPAS=0.3,1,3,10,30,100 $PY p1_snrstart.py $R/diffE_std/last_cfg.pt $O/p1_snrstart_eps_ext.json
ev p1_r3_eps.json $PY p1_r3.py $R/diffE_std/last_cfg.pt $O/p1_r3_eps.json
ev p1_r3_v.json $PY p1_r3.py $R/diffA_std/best.pt $O/p1_r3_v.json
ev p1_r4_eps.json $CPU KAPPA=1.0 RHO_JSON=$O/p1_r3_eps.json $PY p1_r4_cpu.py $R/diffE_std/last_cfg.pt $O/p1_r4_eps.json
# single-stream BER
ev p1_ber_ss.json $EV --out $O/p1_ber_ss.json --methods lmmse --single_stream --train_speed_hi 800 --nets crdm=$R/diffA_std_ss_v800/best.pt,reg=$R/reg_std_ss_v800/best.pt --steps 5,1t --n 1024 --snrs 10,20 --speeds 500 --speed_snr 20 --ber --splits test,unseen
# per-bin spatial rank of the 4x4 channel (Section on the BER setup: why one data stream)
ev rank_stats.json $PY rank_stats.py
# published CRDM checkpoint (Table I, CRDM rows, and the 1000-frame check): needs the original CRDM data and checkpoint
if [ -n "$V1_CKPT" ] && [ -n "$V1_DATA" ]; then
  ev v1_ablation.json bash -c "cd v1check && $PY v1_ablation.py"
  ev v1check_spacing_n1000.json bash -c "cd v1check && $PY v1check.py 1000"
  ev v1check_spacing_n250.json bash -c "cd v1check && $PY v1check.py 250"
else echo "[skip] v1_ablation, v1check: set V1_CKPT and V1_DATA (README)"; fi
echo P1 EVALUATIONS FINISHED; summary

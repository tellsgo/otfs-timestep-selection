"""P1 rebuttal on the published CRDM checkpoint: explicit DDIM timestep sequences (v-prediction, eta=0), same frames
and initial noise for every sequence, paired bootstrap intervals.  lead5 reproduces the diffusers 'leading' S=5 run."""
import json, math, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crdm_model import OTFSDiffusionModel
from diffusers import DDIMScheduler
dev = os.environ.get('OTFS_DEV', 'cuda')
ck = torch.load(os.environ['V1_CKPT'], map_location=dev, weights_only=False)
net = OTFSDiffusionModel(in_channels=8, model_channels=64, channel_mult=(1, 2, 4), num_res_blocks=2,
                         attention_resolutions=(128, 256), dropout=0.1, time_embed_dim=512, num_classes=60,
                         use_class_condition=True, use_explicit_snr=True, use_sinusoidal_snr_embed=True,
                         use_complex_unet=True, use_conditioning=True).to(dev)
sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ck['model_state_dict'].items()}
net.load_state_dict(sd, strict=True); net.eval()
mu, sd_ = ck['residual_mean'], ck['residual_std']
ac = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='squaredcos_cap_v2',
                   prediction_type='v_prediction', clip_sample=False).alphas_cumprod.to(dev)
root = os.environ['V1_DATA']   # the test_seen folder of the CRDM dataset multiscene_otfs_v4
H = np.load(root + '/channels_clean.npy', mmap_mode='r'); Y = np.load(root + '/H_noisy.npy', mmap_mode='r')
L = np.load(root + '/labels.npy'); P = np.load(root + '/power_factors.npy')
SEQ = {'lead5': [800, 600, 400, 200, 0], 'first5': [999, 600, 400, 200, 0], 'trail5': [999, 799, 599, 399, 199],
       'lead3': [666, 333, 0], 'first3': [999, 333, 0], 'trail3': [999, 666, 333]}
PAIRS = [('first5', 'lead5'), ('trail5', 'lead5'), ('trail5', 'first5'), ('first3', 'lead3'), ('trail3', 'lead3'), ('trail3', 'first3'), ('trail3', 'lead5')]
def toR(x): return torch.cat([torch.from_numpy(x.real.copy()), torch.from_numpy(x.imag.copy())], 1).float().to(dev)
def boot(a, b, n=2000):
    g = torch.Generator().manual_seed(0); N = len(a); d = []
    for _ in range(n):
        i = torch.randint(0, N, (N,), generator=g); d.append(10 * math.log10(a[i].mean() / b[i].mean()))
    d = torch.tensor(d); return [10 * math.log10(a.mean() / b.mean()), d.quantile(0.025).item(), d.quantile(0.975).item()]
rng = np.random.default_rng(0); res = {}
for snr in [0, 3, 5, 8, 10]:
    ids = np.where((np.abs(L[:, 2] - snr) < 0.5) & (P > 1e-8))[0]
    ids = np.sort(rng.choice(ids, 1000, replace=False))
    err = {k: [] for k in SEQ}
    for b in range(0, len(ids), 50):
        sl = ids[b:b + 50]
        h = toR(np.array(H[sl])); hl = toR(np.array(Y[sl]))
        spd = torch.tensor(L[sl, 1], device=dev).float(); sn = torch.tensor(L[sl, 2], device=dev).float()
        c = hl / sd_
        g = torch.Generator(device=dev); g.manual_seed(1 + b)
        x0n = torch.randn(hl.shape, generator=g, device=dev)
        for k, ts in SEQ.items():
            x = x0n.clone()
            with torch.no_grad():
                for i, t in enumerate(ts):
                    a = ac[t]; ap = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
                    v = net(c + x, torch.full((x.shape[0],), t, device=dev, dtype=torch.long), speed_kmh=spd, snr_db=sn)
                    x0 = a.sqrt() * x - (1 - a).sqrt() * v; eps = (1 - a).sqrt() * x + a.sqrt() * v
                    x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
            est = hl + x * sd_ + mu
            err[k].append(((est - h).pow(2).flatten(1).sum(1) / h.pow(2).flatten(1).sum(1)).cpu())
    e = {k: torch.cat(v).double() for k, v in err.items()}
    res[f'snr{snr}'] = dict(nmse_db={k: 10 * math.log10(v.mean()) for k, v in e.items()}, diff_db={f'{a}-{b}': boot(e[a], e[b]) for a, b in PAIRS}, n=len(ids))
    print(snr, {k: round(v, 2) for k, v in res[f'snr{snr}']['nmse_db'].items()}, flush=True)
    print('   ', {k: [round(x, 2) for x in v] for k, v in res[f'snr{snr}']['diff_db'].items()}, flush=True)
json.dump(res, open('__OTFS_ROOT__/results/v22/v1_ablation.json', 'w'), indent=1)

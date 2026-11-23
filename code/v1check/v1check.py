"""Re-run the published CRDM checkpoint on its own (v1) test data with two DDIM timestep spacings."""
import json, os, sys, numpy as np, torch
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
print(net.load_state_dict(sd, strict=True)); net.eval()
mu, sd_ = ck['residual_mean'], ck['residual_std']
root = os.environ['V1_DATA']   # the test_seen folder of the CRDM dataset multiscene_otfs_v4
H = np.load(root + '/channels_clean.npy', mmap_mode='r'); Y = np.load(root + '/H_noisy.npy', mmap_mode='r')
L = np.load(root + '/labels.npy'); P = np.load(root + '/power_factors.npy')
rng = np.random.default_rng(0)
res = {}
def toR(x): return torch.cat([torch.from_numpy(x.real.copy()), torch.from_numpy(x.imag.copy())], 1).float().to(dev)
for snr in [0, 3, 5, 8, 10]:
    ids = np.where((np.abs(L[:, 2] - snr) < 0.5) & (P > 1e-8))[0]
    ids = np.sort(rng.choice(ids, int(sys.argv[1]) if len(sys.argv) > 1 else 250, replace=False))
    h = toR(np.array(H[ids])); hl = toR(np.array(Y[ids]))
    spd = torch.tensor(L[ids, 1], device=dev).float(); sn = torch.tensor(L[ids, 2], device=dev).float()
    for spacing in ['leading', 'trailing']:
        for S in [1, 2, 3, 5, 10]:
            sch = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='squaredcos_cap_v2',
                                prediction_type='v_prediction', clip_sample=False, timestep_spacing=spacing)
            sch.set_timesteps(S)
            g = torch.Generator(device=dev); g.manual_seed(1)
            errs = []
            for b in range(0, len(ids), 50):
                x = torch.randn(hl[b:b+50].shape, generator=g, device=dev)
                c = hl[b:b+50] / sd_
                with torch.no_grad():
                    for t in sch.timesteps:
                        tt = torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)
                        out = net(c + x, tt, speed_kmh=spd[b:b+50], snr_db=sn[b:b+50])
                        x = sch.step(out, t, x).prev_sample
                est = hl[b:b+50] + x * sd_ + mu
                e = (est - h[b:b+50]).pow(2).flatten(1).sum(1) / h[b:b+50].pow(2).flatten(1).sum(1)
                errs.append(e)
            v = 10 * torch.log10(torch.cat(errs).mean()).item()
            res[f'snr{snr}_{spacing}_S{S}'] = v
            print(snr, spacing, S, [int(t) for t in sch.timesteps], round(v, 2), flush=True)
json.dump(res, open('__OTFS_ROOT__/results/v22/v1check_spacing_n%s.json' % (sys.argv[1] if len(sys.argv) > 1 else '250'), 'w'), indent=1)

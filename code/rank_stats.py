"""Per-bin spatial rank of the 4x4 channel in the 2D-DFT domain (why four independent streams are not separable)."""
import os
import json, zlib, torch
from otfs5.layouts import get_layout
from otfs5.sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario
dev = os.environ.get('OTFS_DEV', 'cuda'); res = {}
link = Link(layout=get_layout('std'))
for sp, cities, part in (('test', TRAIN_CITIES, 'test'), ('unseen', UNSEEN_CITIES, 'all')):
    scen = Scenario(PathBank('__OTFS_ROOT__/paths', cities, part, device=dev), link)
    svs = []
    for b in range(16):
        g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr(('rank', sp, b)).encode()) % (2**31))
        d = scen.draw(64, g, snr=20.0)
        sv = torch.linalg.svdvals(torch.fft.fft2(d['H']).permute(0, 3, 4, 1, 2))     # [B,N,M,4]
        p = sv.pow(2)
        svs.append((p / p.sum(-1, keepdim=True)).flatten(0, 2)[::7].cpu())         # energy share per singular value
    e = torch.cat(svs)
    res[sp] = dict(mean_energy_share=e.mean(0).tolist(), median_energy_share=e.median(0).values.tolist(),
                   frac_bins_share2_above_1pct=(e[:, 1] > 0.01).float().mean().item(),
                   frac_bins_share3_above_1pct=(e[:, 2] > 0.01).float().mean().item(),
                   frac_bins_share4_above_1pct=(e[:, 3] > 0.01).float().mean().item())
    print(sp, {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in res[sp].items()}, flush=True)
json.dump(res, open('__OTFS_ROOT__/results/v22/rank_stats.json', 'w'), indent=1)

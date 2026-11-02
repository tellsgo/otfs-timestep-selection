"""P1 rebuttal: which part of trailing spacing matters?  Same checkpoint, same frames, explicit timestep sequences:
  lead5 [800,600,400,200,0] (diffusers leading), first5 [999,600,400,200,0] (only the first step moved to the
  pure-noise end), trail5 [999,799,599,399,199]; same for three steps; plus trail1 [999] and the regression network.
Per-frame errors are kept, so differences get paired bootstrap 95% intervals (frames resampled)."""
import os
import json, math, sys, zlib, torch
from otfs5.anchor import AnchorBank
from otfs5.diffusion import alphas_for, ddim_sample, regress
from otfs5.evaluate import load_net
from otfs5.layouts import get_layout
from otfs5.sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db
dev = os.environ.get('OTFS_DEV', 'cuda'); R = '__OTFS_ROOT__/runs_v21'
SEQ = {'lead5': [800, 600, 400, 200, 0], 'first5': [999, 600, 400, 200, 0], 'trail5': [999, 799, 599, 399, 199],
       'lead3': [666, 333, 0], 'first3': [999, 333, 0], 'trail3': [999, 666, 333], 'trail1': [999]}
PAIRS = [('first5', 'lead5'), ('trail5', 'lead5'), ('trail5', 'first5'), ('first3', 'lead3'), ('trail3', 'lead3'),
         ('trail3', 'first3'), ('trail1', 'lead5'), ('reg', 'trail1')]
DIFF = sys.argv[1] if len(sys.argv) > 1 else f'{R}/diffA_std/best.pt'      # optional: another diffusion checkpoint
OUT = sys.argv[2] if len(sys.argv) > 2 else '__OTFS_ROOT__/results/v21/p1_ablation.json'
link = Link(layout=get_layout('std')); anc = AnchorBank(link, 'std')
dn, dsc, dcfg = load_net(DIFF); rn, rsc, _ = load_net(f'{R}/regA_std/best.pt')
PRED = dcfg.get('pred', 'v'); ab = alphas_for(dcfg.get('schedule', 'cosine')).to(dev)

def boot(a, b, n=2000):
    """dB difference 10log10(mean a / mean b) with a paired bootstrap 95% interval over frames."""
    g = torch.Generator().manual_seed(0); N = len(a); d = []
    for _ in range(n):
        i = torch.randint(0, N, (N,), generator=g); d.append(10 * math.log10(a[i].mean() / b[i].mean()))
    d = torch.tensor(d)
    return [10 * math.log10(a.mean() / b.mean()), d.quantile(0.025).item(), d.quantile(0.975).item()]

res = {}
for sp, cities, part in (('test', TRAIN_CITIES, 'test'), ('unseen', UNSEEN_CITIES, 'all')):
    scen = Scenario(PathBank('__OTFS_ROOT__/paths', cities, part, device=dev), link)
    for snr in (0.0, 10.0, 20.0):
        err = {k: [] for k in list(SEQ) + ['reg']}
        for b in range(16):
            g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((sp, 'snr', snr, None, b)).encode()) % (2**31))
            d = scen.draw(64, g, snr=snr); A = anc(d['Hls'], d['snr_db'])
            for k, ts in SEQ.items():
                gm = torch.Generator(device=dev); gm.manual_seed(1000 + b)      # same initial noise for every sequence
                err[k].append(nmse_db(ddim_sample(dn, A, d['snr_db'], dsc, ab, gen=gm, t_start=ts, pred=PRED), d['H'])[1].cpu())
            err['reg'].append(nmse_db(regress(rn, A, d['snr_db'], rsc), d['H'])[1].cpu())
        e = {k: torch.cat(v).double() for k, v in err.items()}
        key = f'{sp}_snr{snr:g}'
        res[key] = dict(nmse_db={k: 10 * math.log10(v.mean()) for k, v in e.items()},
                        diff_db={f'{a}-{b}': boot(e[a], e[b]) for a, b in PAIRS}, n=len(e['reg']))
        print(key, {k: round(v, 2) for k, v in res[key]['nmse_db'].items()}, flush=True)
        print('   ', {k: [round(x, 2) for x in v] for k, v in res[key]['diff_db'].items()}, flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

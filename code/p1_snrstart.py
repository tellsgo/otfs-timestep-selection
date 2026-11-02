"""P1 round 2 (fresh reviewer): a timestep selection driven by the pilot observation.  The state x_t = sqrt(a_t) r +
sqrt(1 - a_t) e carries the information ratio q_t = a_t v / (1 - a_t) about the normalised anchor residual r, whose
per-element variance v(snr) follows from the anchor error at the operating SNR (validation frames).  SNR-matched start:
t* = min{t : q_t <= kappa}, initial state x_{t*} ~ N(0, a_{t*} v + 1 - a_{t*}) (the exact marginal variance), then S
DDIM steps on [t*, t*(S-1)/S, ..., t*/S].  kappa -> 0 gives t* = T - 1 and the trailing grid.  kappa is chosen once per
checkpoint on validation frames at S = 3; the sweep is stored.  Compared with leading, first-only and trailing grids
(same checkpoint, same frames, same initial noise) and the regression network (one forward pass).
  python p1_snrstart.py DIFF_CKPT OUT.json"""
import json, math, sys, zlib, torch
from otfs5.anchor import AnchorBank
from otfs5.diffusion import alphas_for, ddim_sample, regress, to_real
from otfs5.evaluate import load_net
from otfs5.layouts import get_layout
from otfs5.sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db
dev = os.environ.get('OTFS_DEV', 'cuda'); R = '__OTFS_ROOT__/runs_v21'; P = '__OTFS_ROOT__/paths'; T = 1000
DIFF, OUT = sys.argv[1], sys.argv[2]
link = Link(layout=get_layout('std')); anc = AnchorBank(link, 'std')
dn, dsc, dcfg = load_net(DIFF); rn, rsc, _ = load_net(f'{R}/regA_std/best.pt')
PRED = dcfg.get('pred', 'v'); ab = alphas_for(dcfg.get('schedule', 'cosine')).to(dev)
import os
KAPPAS = tuple(float(k) for k in os.environ['KAPPAS'].split(',')) if 'KAPPAS' in os.environ else \
    (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)   # 0 gives t* = T - 1, the trailing grid
GRID = {'lead': lambda S: [int(t) for t in range((S - 1) * (T // S), -1, -(T // S))],
        'first': lambda S: [T - 1] + [int(t) for t in range((S - 1) * (T // S), -1, -(T // S))][1:] if S > 1 else [T - 1],
        'trail': lambda S: [int(round(T - i * T / S)) - 1 for i in range(S)]}

def boot(a, b, n=2000):
    g = torch.Generator().manual_seed(0); N = len(a); d = []
    for _ in range(n):
        i = torch.randint(0, N, (N,), generator=g); d.append(10 * math.log10(a[i].mean() / b[i].mean()))
    d = torch.tensor(d)
    return [10 * math.log10(a.mean() / b.mean()), d.quantile(0.025).item(), d.quantile(0.975).item()]

def frames(scen, tag, snr, b):
    g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((tag, 'snr', snr, None, b)).encode()) % (2**31))
    d = scen.draw(64, g, snr=snr); return d, anc(d['Hls'], d['snr_db'])

def start(v, kappa):
    q = ab * v / (1 - ab)                                  # decreasing in t
    ok = (q <= kappa).nonzero()
    return int(ok[0]) if len(ok) else T - 1

def snr_grid(ts_star, S):
    return sorted({max(1, int(round(ts_star * (S - i) / S))) for i in range(S)}, reverse=True)

@torch.no_grad()
def run_snr(A, d, ts, b, v=None):
    gm = torch.Generator(device=dev); gm.manual_seed(1000 + b)
    x = torch.randn(to_real(A).shape, generator=gm, device=dev)
    if v is not None:
        x = x * (ab[ts[0]] * v + 1 - ab[ts[0]]).sqrt()
    return nmse_db(ddim_sample(dn, A, d['snr_db'], dsc, ab, x_T=x, t_start=ts, pred=PRED), d['H'])[1].cpu()

with torch.no_grad():
    # residual variance of the anchor per SNR, validation users of the training cities
    val = Scenario(PathBank(P, TRAIN_CITIES, 'val', device=dev), link); V = {}
    for snr in (0.0, 10.0, 20.0):
        acc = []
        for b in range(4):
            d, A = frames(val, 'v_val', snr, b); acc.append((to_real(d['H'] - A) / dsc).pow(2).mean().item())
        V[snr] = sum(acc) / len(acc)
    print('residual variance v(snr)', {k: f'{x:.4g}' for k, x in V.items()}, flush=True)
    # kappa sweep on validation frames, S = 1, 3, 5
    sweep = {}
    for S in (1, 3, 5):
        for k in KAPPAS:
            e = []
            for snr in (0.0, 10.0, 20.0):
                ts = snr_grid(start(V[snr], k), S)
                for b in range(4):
                    d, A = frames(val, 's_val', snr, b); e.append(10 * math.log10(run_snr(A, d, ts, b, V[snr]).mean()))
            sweep[f'S{S}_k{k:g}'] = sum(e) / len(e)
        print('sweep', S, {kk: round(x, 2) for kk, x in sweep.items() if kk.startswith(f'S{S}_')}, flush=True)
    kap = min(KAPPAS, key=lambda k: sweep[f'S3_k{k:g}'])
    starts = {snr: start(V[snr], kap) for snr in V}
    print('kappa', kap, 'starts', starts, flush=True)
    res = dict(v=V, sweep=sweep, kappa=kap, starts=starts, pred=PRED, schedule=dcfg.get('schedule', 'cosine'))
    for sp, cities, part in (('test', TRAIN_CITIES, 'test'), ('unseen', UNSEEN_CITIES, 'all')):
        scen = Scenario(PathBank(P, cities, part, device=dev), link)
        for snr in (0.0, 10.0, 20.0):
            err = {}
            for b in range(16):
                d, A = frames(scen, sp, snr, b)
                for S in (1, 3, 5):
                    for g, f in GRID.items():
                        err.setdefault(f'{g}{S}', []).append(run_snr(A, d, f(S), b))
                    err.setdefault(f'snr{S}', []).append(run_snr(A, d, snr_grid(starts[snr], S), b, V[snr]))
                err.setdefault('reg', []).append(nmse_db(regress(rn, A, d['snr_db'], rsc), d['H'])[1].cpu())
            e = {k: torch.cat(x).double() for k, x in err.items()}
            pairs = [(f'snr{S}', f'{g}{S}') for S in (1, 3, 5) for g in GRID] + [(f'snr{S}', 'reg') for S in (1, 3, 5)]
            key = f'{sp}_snr{snr:g}'
            res[key] = dict(nmse_db={k: 10 * math.log10(x.mean()) for k, x in e.items()},
                            diff_db={f'{a}-{b}': boot(e[a], e[b]) for a, b in pairs}, n=len(e['reg']))
            print(key, {k: round(x, 2) for k, x in res[key]['nmse_db'].items()}, flush=True)
            print('   ', {k: [round(x, 2) for x in y] for k, y in res[key]['diff_db'].items()}, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

"""P1 round 3: is the SNR-matched start better than a tuned interior start, and can it be made per frame?
One step (S = 1) on the epsilon-prediction network, initial state with the matched marginal variance in every case.
  fixed      : one start t_s for all SNRs, chosen on validation frames from a grid
  per_snr    : one start per SNR, chosen on validation frames (three free parameters)
  matched    : t* from kappa (one parameter, chosen on validation), v(SNR) from validation frames (as p1_snrstart.py)
  frame      : as matched with v scaled per frame by the pilot-window residual of the anchor, rho / mean(rho)
Also the v-prediction network with the same four rules.  Same frames and noise as p1_snrstart.py.
  python p1_r3.py CKPT OUT.json"""
import os
import json, math, sys, zlib, torch
from otfs5.anchor import AnchorBank
from otfs5.diffusion import alphas_for, ddim_sample, to_real
from otfs5.evaluate import load_net
from otfs5.layouts import get_layout
from otfs5.sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db
dev = os.environ.get('OTFS_DEV', 'cuda'); P = '__OTFS_ROOT__/paths'; T = 1000
CK, OUT = sys.argv[1], sys.argv[2]
link = Link(layout=get_layout('std')); anc = AnchorBank(link, 'std', snrs=(0.0, 10.0, 20.0))
dn, dsc, dcfg = load_net(CK); PRED = dcfg.get('pred', 'v'); ab = alphas_for(dcfg.get('schedule', 'cosine')).to(dev)
wmask = link.window_mask(dev)[None, None]
TS = (25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 900, 950, 999)
KAPPAS = (1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0)

def frames(scen, tag, snr, b):
    g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((tag, 'snr', snr, None, b)).encode()) % (2**31))
    d = scen.draw(64, g, snr=snr); return d, anc(d['Hls'], d['snr_db'])

def rho(d, A):                                   # pilot-window residual of the anchor per frame
    return ((d['Hls'] - A) * wmask).abs().pow(2).flatten(1).sum(1)

def tstar(v, kappa):                             # v: tensor [B]; returns per-frame start
    q = ab[None] * v[:, None] / (1 - ab[None])
    ok = q <= kappa
    t = torch.where(ok.any(1), ok.float().argmax(1), torch.full_like(v, T - 1, dtype=torch.long))
    return t.clamp(1, T - 1)

@torch.no_grad()
def one_step(A, d, t, v, b):
    """one DDIM step from start t (scalar or per frame), initial state with the matched marginal variance."""
    gm = torch.Generator(device=dev); gm.manual_seed(1000 + b)
    x = torch.randn(to_real(A).shape, generator=gm, device=dev)
    t = torch.as_tensor(t, device=dev).expand(A.shape[0]) if not torch.is_tensor(t) or t.dim() == 0 else t
    out = torch.empty_like(A)
    for tv in t.unique():
        m = t == tv; a = ab[tv]
        xs = x[m] * (a * v[m] + 1 - a).sqrt().view(-1, *([1] * (x.dim() - 1)))
        out[m] = ddim_sample(dn, A[m], d['snr_db'][m], dsc, ab, x_T=xs, t_start=[int(tv)], pred=PRED)
    return out

res = {}
with torch.no_grad():
    val = Scenario(PathBank(P, TRAIN_CITIES, 'val', device=dev), link); V = {}; RB = {}
    for snr in (0.0, 10.0, 20.0):
        acc, rr = [], []
        for b in range(4):
            d, A = frames(val, 'v_val', snr, b); acc.append((to_real(d['H'] - A) / dsc).pow(2).mean().item()); rr.append(rho(d, A))
        V[snr] = sum(acc) / len(acc); RB[snr] = torch.cat(rr).mean()
    # validation sweeps
    sw = {}
    for snr in (0.0, 10.0, 20.0):
        for b in range(4):
            d, A = frames(val, 's_val', snr, b); v0 = torch.full((A.shape[0],), V[snr], device=dev); vf = v0 * rho(d, A) / RB[snr]
            for t in TS:
                sw.setdefault(('t', t, snr), []).append(nmse_db(one_step(A, d, t, v0, b), d['H'])[1])
            for k in KAPPAS:
                sw.setdefault(('k', k, snr), []).append(nmse_db(one_step(A, d, tstar(v0, k), v0, b), d['H'])[1])
                sw.setdefault(('f', k, snr), []).append(nmse_db(one_step(A, d, tstar(vf, k), vf, b), d['H'])[1])
    db = {k: 10 * math.log10(torch.cat(v).mean().item()) for k, v in sw.items()}
    avg = lambda kind, p: sum(db[(kind, p, s)] for s in (0.0, 10.0, 20.0)) / 3
    t_fixed = min(TS, key=lambda t: avg('t', t)); k_m = min(KAPPAS, key=lambda k: avg('k', k)); k_f = min(KAPPAS, key=lambda k: avg('f', k))
    t_snr = {s: min(TS, key=lambda t: db[('t', t, s)]) for s in (0.0, 10.0, 20.0)}
    res['val'] = {f'{a}_{p:g}_snr{s:g}': x for (a, p, s), x in db.items()}
    res['selected'] = dict(t_fixed=t_fixed, t_per_snr=t_snr, kappa_matched=k_m, kappa_frame=k_f, v=V)
    print('selected', res['selected'], flush=True)
    for sp, cities, part in (('test', TRAIN_CITIES, 'test'), ('unseen', UNSEEN_CITIES, 'all')):
        scen = Scenario(PathBank(P, cities, part, device=dev), link)
        for snr in (0.0, 10.0, 20.0):
            e = {k: [] for k in ('fixed', 'per_snr', 'matched', 'frame')}; starts = []
            for b in range(16):
                d, A = frames(scen, sp, snr, b); v0 = torch.full((A.shape[0],), V[snr], device=dev); vf = v0 * rho(d, A) / RB[snr]
                e['fixed'].append(nmse_db(one_step(A, d, t_fixed, v0, b), d['H'])[1])
                e['per_snr'].append(nmse_db(one_step(A, d, t_snr[snr], v0, b), d['H'])[1])
                e['matched'].append(nmse_db(one_step(A, d, tstar(v0, k_m), v0, b), d['H'])[1])
                tf = tstar(vf, k_f); starts.append(tf.float().cpu())
                e['frame'].append(nmse_db(one_step(A, d, tf, vf, b), d['H'])[1])
            e = {k: torch.cat(v).double().cpu() for k, v in e.items()}
            def boot(a, c, n=2000):
                g = torch.Generator().manual_seed(0); N = len(a); s = []
                for _ in range(n):
                    i = torch.randint(0, N, (N,), generator=g); s.append(10 * math.log10(a[i].mean().item() / c[i].mean().item()))
                s = torch.tensor(s); return [10 * math.log10(a.mean().item() / c.mean().item()), s.quantile(0.025).item(), s.quantile(0.975).item()]
            st = torch.cat(starts)
            res[f'{sp}_snr{snr:g}'] = dict(nmse_db={k: 10 * math.log10(v.mean().item()) for k, v in e.items()},
                                           diff_db={f'{a}-fixed': boot(e[a], e['fixed']) for a in ('per_snr', 'matched', 'frame')} | {'frame-matched': boot(e['frame'], e['matched'])},
                                           frame_start=dict(mean=st.mean().item(), p10=st.quantile(0.1).item(), p90=st.quantile(0.9).item()))
            print(sp, snr, {k: round(v, 2) for k, v in res[f'{sp}_snr{snr:g}']['nmse_db'].items()}, {k: [round(x, 2) for x in v] for k, v in res[f'{sp}_snr{snr:g}']['diff_db'].items()}, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

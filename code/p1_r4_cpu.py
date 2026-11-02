"""P1 round 4 (from p1_r3.py, CPU): prediction at SNRs not used for calibration.  v(SNR) at 5 and 15 dB is interpolated
log-linearly from the validation values at 0, 10 and 20 dB (no labels at 5 or 15 dB), kappa is the value chosen before
(1 for the epsilon network), and the predicted start is compared with the best start on a fine grid evaluated on the test
frames (an oracle) and with the single tuned start t = 300.  Round 3 header:
P1 round 3: is the SNR-matched start better than a tuned interior start, and can it be made per frame?
One step (S = 1) on the epsilon-prediction network, initial state with the matched marginal variance in every case.
  fixed      : one start t_s for all SNRs, chosen on validation frames from a grid
  per_snr    : one start per SNR, chosen on validation frames (three free parameters)
  matched    : t* from kappa (one parameter, chosen on validation), v(SNR) from validation frames (as p1_snrstart.py)
  frame      : as matched with v scaled per frame by the pilot-window residual of the anchor, rho / mean(rho)
Also the v-prediction network with the same four rules.  Same frames and noise as p1_snrstart.py.
  python p1_r3.py CKPT OUT.json"""
import json, math, sys, zlib, torch
from otfs5.anchor import AnchorBank
from otfs5.diffusion import alphas_for, ddim_sample, to_real
from otfs5.evaluate import load_net
from otfs5.layouts import get_layout
from otfs5.sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db
import os; dev = os.environ.get('OTFS_DEV', 'cuda'); torch.set_num_threads(int(os.environ.get('OTFS_THREADS', '16'))); P = '__OTFS_ROOT__/paths'; T = 1000
CK, OUT = sys.argv[1], sys.argv[2]
link = Link(layout=get_layout('std')); anc = AnchorBank(link, 'std', snrs=(0.0, 5.0, 10.0, 15.0, 20.0), device=dev)
dn, dsc, dcfg = load_net(CK, dev=dev); PRED = dcfg.get('pred', 'v'); ab = alphas_for(dcfg.get('schedule', 'cosine')).to(dev)
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
TSF = (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500)
with torch.no_grad():
    val = Scenario(PathBank(P, TRAIN_CITIES, 'val', device=dev), link); V = {}
    for snr in (0.0, 10.0, 20.0):
        acc = []
        for b in range(4):
            d, A = frames(val, 'v_val', snr, b); acc.append((to_real(d['H'] - A) / dsc).pow(2).mean().item())
        V[snr] = sum(acc) / len(acc)
    lv = {s_: math.log(V[s_]) for s_ in V}
    interp = lambda x: math.exp(lv[0.0] + (lv[10.0] - lv[0.0]) * x / 10) if x <= 10 else math.exp(lv[10.0] + (lv[20.0] - lv[10.0]) * (x - 10) / 10)
    KAP = float(os.environ.get('KAPPA', '1.0'))
    res['v'] = V; res['kappa'] = KAP
    for sp, cities, part in (('test', TRAIN_CITIES, 'test'), ('unseen', UNSEEN_CITIES, 'all')):
        scen = Scenario(PathBank(P, cities, part, device=dev), link)
        for snr in (5.0, 15.0):
            vi = interp(snr); tp = int(tstar(torch.tensor([vi], device=dev), KAP)[0])
            starts = sorted(set(TSF) | {tp})
            e = {t: [] for t in starts}
            for b in range(16):
                d, A = frames(scen, sp, snr, b); v0 = torch.full((A.shape[0],), vi, device=dev)
                for t in starts:
                    e[t].append(nmse_db(one_step(A, d, t, v0, b), d['H'])[1])
            e = {t: torch.cat(x).double().cpu() for t, x in e.items()}
            db = {t: 10 * math.log10(x.mean().item()) for t, x in e.items()}
            tb = min(db, key=db.get)
            def boot(a, c, n=2000):
                g = torch.Generator().manual_seed(0); N = len(a); s_ = []
                for _ in range(n):
                    i = torch.randint(0, N, (N,), generator=g); s_.append(10 * math.log10(a[i].mean().item() / c[i].mean().item()))
                s_ = torch.tensor(s_); return [10 * math.log10(a.mean().item() / c.mean().item()), s_.quantile(0.025).item(), s_.quantile(0.975).item()]
            res[f'{sp}_snr{snr:g}'] = dict(v_interp=vi, t_pred=tp, t_best=tb, nmse_db={str(t): x for t, x in db.items()},
                                          pred_minus_best=boot(e[tp], e[tb]), pred_minus_fixed300=boot(e[tp], e[300]))
            print(sp, snr, 'v', round(vi, 4), 't_pred', tp, 't_best', tb, 'pred', round(db[tp], 2), 'best', round(db[tb], 2), 'fixed300', round(db[300], 2),
                  [round(x, 2) for x in res[f'{sp}_snr{snr:g}']['pred_minus_fixed300']], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

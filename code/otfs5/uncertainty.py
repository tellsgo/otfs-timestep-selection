"""Posterior-sample uncertainty of the anchored diffusion estimator and its use in detection.

Mean:      one trailing DDIM step (optionally averaged over seeds).
Variance:  spread of K multi-step posterior samples, mapped to the 2D-DFT domain.
Baselines: input-independent error spectra of the same estimator and of the LMMSE anchor,
           measured on training frames at the same SNR.
"""
import argparse
import zlib
import json
import math
import os

import torch

from .anchor import AnchorBank
from .baselines import detect_ber, detect_ber_cov
from .diffusion import cosine_alphas_cumprod, ddim_sample, regress
from .evaluate import load_net
from .layouts import get_layout
from .sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario


def sample_set(net, A, snr, sc, ab, K, S, gen):
    return torch.stack([ddim_sample(net, A, snr, sc, ab, S=S, gen=gen, spacing='trailing') for _ in range(K)])


def spec(E):
    """Per-bin power of an error tensor in the 2D-DFT domain, averaged over the batch: [nr,nt,N,M]."""
    return torch.fft.fft2(E).abs().pow(2).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diff', required=True)
    p.add_argument('--reg', default=None)
    p.add_argument('--layout', default='std')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--S', type=int, default=20)
    p.add_argument('--snrs', default='0,5,10,15,20')
    p.add_argument('--n_stat', type=int, default=1024)
    p.add_argument('--paths', default='__OTFS_ROOT__/paths')
    p.add_argument('--single_stream', action='store_true')
    p.add_argument('--train_speed_hi', type=float, default=500.0, help='speed range of the anchor statistics')
    p.add_argument('--speed_hi', type=float, default=500.0, help='speed range of the evaluated frames')
    a = p.parse_args()
    dev = os.environ.get('OTFS_DEV', 'cuda')
    link = Link(layout=get_layout(a.layout), data_tx=(0,) if a.single_stream else None)
    anc = AnchorBank(link, a.layout, speed_range=(30.0, a.train_speed_hi))
    streams = link.data_tx
    ab = cosine_alphas_cumprod().to(dev)
    net, sc, _ = load_net(a.diff)
    reg = load_net(a.reg) if a.reg else None
    train = Scenario(PathBank(a.paths, TRAIN_CITIES, 'train', device=dev), link, (30.0, a.speed_hi))
    res = {}
    for split in ('test', 'unseen'):
        cities = TRAIN_CITIES if split == 'test' else UNSEEN_CITIES
        scen = Scenario(PathBank(a.paths, cities, 'test' if split == 'test' else 'all', device=dev), link, (30.0, a.speed_hi))
        for snr in [float(x) for x in a.snrs.split(',')]:
            # input-independent error spectra, measured on training frames
            g = torch.Generator(device=dev); g.manual_seed(77)
            sp_d = sp_a = sp_r = 0
            nb = a.n_stat // 64
            tr_post = tr_err = 0.0          # training-frame totals for the frame factor and the global calibration
            Cf = 0; pres = 0.0               # receive-antenna error covariance per bin; mean pilot-window residual (online severity)
            wmask = link.window_mask(dev)[None, None]
            for ib in range(nb):
                d = train.draw(64, g, snr=snr)
                A = anc(d['Hls'], d['snr_db'])
                sp_a = sp_a + spec(A - d['H']) / nb
                Hm = ddim_sample(net, A, d['snr_db'], sc, ab, S=1, gen=g, spacing='trailing')
                sp_d = sp_d + spec(Hm - d['H']) / nb
                if reg:
                    Hr_t = regress(reg[0], A, d['snr_db'], reg[1])
                    sp_r = sp_r + spec(Hr_t - d['H']) / nb
                    Df = torch.fft.fft2(Hr_t - d['H'])[:, :, 0]                               # [B,r,N,M], data antenna 0
                    Cf = Cf + torch.einsum('brkl,bskl->klrs', Df, Df.conj()) / (Df.shape[0] * nb)
                    pres = pres + ((d['Hls'] - Hr_t) * wmask).abs().pow(2).flatten(1).sum(1).mean().item() / nb
                if ib < max(1, nb // 4):    # posterior samples are costly: a quarter of the frames
                    Hs = sample_set(net, A, d['snr_db'], sc, ab, a.K, a.S, g)
                    tr_post += ((Hs - Hs.mean(0)).abs().pow(2).sum(0) / (a.K - 1)).sum().item()
                    tr_err += (Hm - d['H']).abs().pow(2).sum().item()
            sp_site = None
            if reg and split == 'unseen':    # spectrum re-measured on other frames of the unseen city
                g2 = torch.Generator(device=dev); g2.manual_seed(78); sp_site = 0
                for _ in range(nb):
                    d = scen.draw(64, g2, snr=snr); A = anc(d['Hls'], d['snr_db'])
                    sp_site = sp_site + spec(regress(reg[0], A, d['snr_db'], reg[1]) - d['H']) / nb
            n_post = max(1, nb // 4) * 64
            mean_post = tr_post / n_post                 # mean per-frame posterior variance (total)
            c_cal = tr_err / tr_post                     # global calibration factor of the posterior variance
            acc = {}
            cal = []
            for b in range(a.n // 64):
                g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((split, snr, b)).encode()) % (2**31))
                d = scen.draw(64, g, snr=snr)
                A = anc(d['Hls'], d['snr_db'])
                Hm = ddim_sample(net, A, d['snr_db'], sc, ab, S=1, gen=g, spacing='trailing')
                Hs = sample_set(net, A, d['snr_db'], sc, ab, a.K, a.S, g)                 # [K,B,...]
                mu = Hs.mean(0)
                var_dd = (Hs - mu).abs().pow(2).sum(0) / (a.K - 1)                        # per element, DD
                Vs = torch.fft.fft2(Hs - mu).abs().pow(2).sum(0) / (a.K - 1)              # per bin, 2D-DFT
                err = (Hm - d['H']).abs().pow(2)
                cal.append(torch.stack([var_dd.flatten(), err.flatten()], 1)[::97])
                f = (var_dd.flatten(1).sum(1) / mean_post)[:, None, None, None, None]   # per-frame severity
                flat = lambda sp: sp.mean((-1, -2), keepdim=True).expand_as(sp)[None].expand_as(Vs)   # scalar NMSE-aware
                runs = {
                    'perfect': (d['H'], None),
                    'lmmse': (A, None), 'lmmse+flat': (A, flat(sp_a)), 'lmmse+stat': (A, sp_a[None].expand_as(Vs)),
                    'diff': (Hm, None), 'diff+flat': (Hm, flat(sp_d)), 'diff+stat': (Hm, sp_d[None].expand_as(Vs)), 'diff+post': (Hm, Vs),
                    'diff+post_cal': (Hm, c_cal * Vs), 'diff+stat*frame': (Hm, sp_d[None] * f),
                }
                if reg:
                    Hr = regress(reg[0], A, d['snr_db'], reg[1])
                    fp = ((d['Hls'] - Hr) * wmask).abs().pow(2).flatten(1).sum(1)[:, None, None, None, None] / pres  # online severity
                    runs.update({'reg': (Hr, None), 'reg+flat': (Hr, flat(sp_r)), 'reg+stat': (Hr, sp_r[None].expand_as(Vs)),
                                 'reg+stat*frame': (Hr, sp_r[None] * f), 'reg+stat*pilot': (Hr, sp_r[None] * fp)})
                    if sp_site is not None:
                        runs['reg+stat_site'] = (Hr, sp_site[None].expand_as(Vs))
                    e, n = detect_ber_cov(link, d['Y'], Hr, d['X'], d['bits'], d['guard'], d['snr_db'], Cf, sp_r)
                    o = acc.setdefault('reg+rxcov', [0, 0]); o[0] += e; o[1] += n
                for k, (Hh, V) in runs.items():
                    e, n = detect_ber(link, d['Y'], Hh, d['X'], d['bits'], d['guard'], d['snr_db'], V, streams=streams)
                    o = acc.setdefault(k, [0, 0]); o[0] += e; o[1] += n
                # frame-level calibration: predicted vs actual total error
                o = acc.setdefault('_frame', [])
                o += list(zip(var_dd.flatten(1).sum(1).tolist(), err.flatten(1).sum(1).tolist()))
            fr = torch.tensor(acc.pop('_frame'))
            rank = lambda x: x.argsort().argsort().float()
            rho = torch.corrcoef(torch.stack([rank(fr[:, 0]), rank(fr[:, 1])]))[0, 1].item()
            cal = torch.cat(cal).cpu()
            q = torch.quantile(cal[:, 0], torch.linspace(0, 1, 11))
            bins = [[cal[(cal[:, 0] >= q[i]) & (cal[:, 0] <= q[i + 1]), j].mean().item() for j in (0, 1)] for i in range(10)]
            res[f'{split}_snr{snr:g}'] = dict(ber={k: v[0] / v[1] for k, v in acc.items()}, spearman_frame=rho,
                                              c_cal=c_cal,
                                              calibration_deciles=bins)
            print(split, snr, {k: f'{v[0]/v[1]:.2e}' for k, v in acc.items()}, 'rho', round(rho, 3), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

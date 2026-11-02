"""Fixed-seed evaluation of estimators: NMSE and uncoded BER over SNR and speed grids."""
import argparse
import zlib
import json
import os
import time

import torch

from .anchor import AnchorBank, make_anchor
from .baselines import GSOMP, LMMSE, SOMP, detect_ber, threshold_estimator
from .diffusion import build_unet, cosine_alphas_cumprod, ddim_sample, regress
from .layouts import get_layout
from .sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db


def load_net(path, dev=os.environ.get('OTFS_DEV', 'cuda')):
    s = torch.load(path, map_location=dev)
    c = s['config']
    cm = tuple(int(x) for x in c['channel_mult'].split(','))
    att = tuple(int(x) for x in c['attention'].split(',')) if c['attention'] else ()
    net = build_unet(c['model_channels'], cm, att, num_res_blocks=c.get('num_res_blocks', 2)).to(dev)
    net.load_state_dict(s['ema']); net.eval()
    return net, s['scale'], c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--layout', default='std')
    p.add_argument('--nets', default='', help='name=path[,name=path]; mode and anchor are read from the checkpoint')
    p.add_argument('--steps', default='5')
    p.add_argument('--methods', default='ls,thr,gsomp,lmmse')
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--snrs', default='0,5,10,15,20')
    p.add_argument('--speeds', default='60,120,250,375,500')
    p.add_argument('--speed_snr', type=float, default=10.0)
    p.add_argument('--ber', action='store_true')
    p.add_argument('--splits', default='test,unseen')
    p.add_argument('--paths', default='__OTFS_ROOT__/paths')
    p.add_argument('--centre', action='store_true')
    p.add_argument('--single_stream', action='store_true', help='only antenna 0 carries data')
    p.add_argument('--train_speed_hi', type=float, default=500.0, help='speed range of the anchor statistics')
    p.add_argument('--anchor_snrs', default=None, help='load the LMMSE anchor only at these grid SNRs (GPU memory)')
    a = p.parse_args()
    dev = os.environ.get('OTFS_DEV', 'cuda')
    lay = get_layout(a.layout)
    link = Link(layout=lay, nt=lay.nt, data_tx=(0,) if a.single_stream else None)
    methods = a.methods.split(',')
    ab = cosine_alphas_cumprod().to(dev)
    nets = {}
    for spec in filter(None, a.nets.split(',')):
        k, v = spec.split('=')
        nets[k] = load_net(v)
    steps = a.steps.split(',')
    somp = SOMP(link) if 'somp' in methods else None
    gsomp = GSOMP(link) if 'gsomp' in methods else None
    akw = {} if a.anchor_snrs is None else dict(snrs=tuple(float(x) for x in a.anchor_snrs.split(',')))
    anc = AnchorBank(link, a.layout, centre=a.centre, speed_range=(30.0, a.train_speed_hi), **akw)
    ancs = {}

    def est(name, d, gen):
        if name == 'ls': return d['Hls']
        if name == 'lmmse': return anc(d['Hls'], d['snr_db'])
        if name.startswith('lmmse_'):                 # structured anchors alone, e.g. lmmse_rx0.25
            k = name[6:]
            if k not in ancs: ancs[k] = make_anchor(k, link, a.layout, speed_range=(30.0, a.train_speed_hi))
            return ancs[k](d['Hls'], d['snr_db'])
        if name == 'thr': return threshold_estimator(link, d['Hls'], d['snr_db'])
        if name == 'somp': return somp(d['Hls'], d['snr_db'])
        if name == 'gsomp': return gsomp(d['Y'], d['snr_db'])
        base, _, S = name.partition('@')
        net, sc, cfg = nets[base]
        kind = cfg.get('anchor', 'ls')
        if kind not in ('ls', 'lmmse') and kind not in ancs:
            ancs[kind] = make_anchor(kind, link, a.layout, speed_range=(30.0, a.train_speed_hi))
        A = d['Hls'] if kind == 'ls' else anc(d['Hls'], d['snr_db']) if kind == 'lmmse' else ancs[kind](d['Hls'], d['snr_db'])
        if cfg['mode'] == 'reg':
            return regress(net, A, d['snr_db'], sc)
        # S may carry a spacing tag and a sample count: '5', '5t' (trailing), '5t:8' (mean of 8 seeds)
        S, _, K = S.partition(':')
        sp = 'trailing' if S.endswith('t') else 'leading'
        S = int(S.rstrip('t'))
        K = int(K) if K else 1
        out = 0
        for _ in range(K):
            out = out + ddim_sample(net, A, d['snr_db'], sc, ab, S=S, gen=gen, spacing=sp)
        return out / K

    names = list(methods)
    for k, (_, _, cfg) in nets.items():
        names += [k] if cfg['mode'] == 'reg' else [f'{k}@{S}' for S in steps]
    res = dict(layout=a.layout, overhead=link.layout.overhead(64, 128), n=a.n, results={})
    for split in a.splits.split(','):
        cities = TRAIN_CITIES if split == 'test' else UNSEEN_CITIES
        scen = Scenario(PathBank(a.paths, cities, 'test' if split == 'test' else 'all', device=dev), link, centre=a.centre)
        R = res['results'].setdefault(split, {})
        grid = [('snr', float(s), None) for s in a.snrs.split(',')] + \
               [('speed', a.speed_snr, float(v)) for v in a.speeds.split(',')]
        for kind, snr, spd in grid:
            key = f'{kind}={spd if kind=="speed" else snr}'
            acc = {m: dict(err=[], be=0, nb=0, t=0.0) for m in names}
            for b in range(a.n // 64):
                g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((split, kind, snr, spd, b)).encode()) % (2**31))
                d = scen.draw(64, g, speed=spd, snr=snr)
                for m in names:
                    gm = torch.Generator(device=dev); gm.manual_seed(1000 + b)
                    (torch.cuda.synchronize() if dev == "cuda" else None); t0 = time.time()
                    Hh = est(m, d, gm)
                    (torch.cuda.synchronize() if dev == "cuda" else None); acc[m]['t'] += time.time() - t0
                    acc[m]['err'].append(nmse_db(Hh, d['H'])[1])
                    if a.ber:
                        e, n = detect_ber(link, d['Y'], Hh, d['X'], d['bits'], d['guard'], d['snr_db'], streams=link.data_tx)
                        acc[m]['be'] += e; acc[m]['nb'] += n
                if a.ber and b == 0:
                    acc.setdefault('perfect', dict(err=[], be=0, nb=0, t=0.0))
                if a.ber:
                    e, n = detect_ber(link, d['Y'], d['H'], d['X'], d['bits'], d['guard'], d['snr_db'], streams=link.data_tx)
                    acc['perfect']['be'] += e; acc['perfect']['nb'] += n
            out = {}
            for m, v in acc.items():
                o = {}
                if v['err']:
                    e = torch.cat(v['err'])
                    o['nmse_db'] = 10 * torch.log10(e.mean()).item()
                    o['ms_per_frame'] = 1e3 * v['t'] / a.n
                if v['nb']:
                    o['ber'] = v['be'] / v['nb']
                out[m] = o
            R[key] = out
            print(split, key, {m: round(o.get('nmse_db', float('nan')), 2) for m, o in out.items()},
                  {m: o['ber'] for m, o in out.items() if 'ber' in o} if a.ber else '', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

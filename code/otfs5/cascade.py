"""Data-aided cascade: stage 1 = anchored regressor on the pilot windows; stage k >= 2 cancels the
data decided with stage k-1, reads wider windows, applies its own LMMSE anchor and regressor.

  python -m otfs5.cascade train --out RUN --stage1 S1.pt [--prev S2_dir,...] --kmax2 14
  python -m otfs5.cascade eval  --out RES.json --stage1 S1.pt --prev S2_dir,S3_dir --kmax2 14
"""
import argparse
import copy
import json
import math
import os
import time
import zlib

import torch

from .anchor import SNR_GRID, AnchorBank, half_cmatmul
from .baselines import detect_ber
from .diffusion import build_unet, regress, regression_loss, to_real
from .evaluate import load_net
from .layouts import get_layout
from .sim import TRAIN_CITIES, UNSEEN_CITIES, Link, PathBank, Scenario, nmse_db
from .twostage import GenericLMMSE, decide, stage2_input, wide_link

PATHS = '__OTFS_ROOT__/paths'


class StageAnchors:
    """LMMSE anchors of one cascade stage on the SNR grid, built from the chain of earlier stages."""

    def __init__(self, root, producer_for_snr, link_obs, device=os.environ.get('OTFS_DEV', 'cuda')):
        self.W, self.snrs = {}, SNR_GRID
        for s in self.snrs:
            f = f'{root}/stage_anchor_snr{s:+.1f}.pt'
            if os.path.exists(f):
                W = torch.load(f, map_location='cpu')
            else:
                W = GenericLMMSE(producer_for_snr(s), link_obs).W.cpu()
                torch.save(W, f + '.tmp'); os.replace(f + '.tmp', f)
                torch.cuda.empty_cache()
            self.W[s] = (W.real.to(device, torch.float16), W.imag.to(device, torch.float16))
            del W
        self.link_obs = link_obs

    def __call__(self, Hobs, snr_db):
        from .baselines import window_vectors
        y = window_vectors(self.link_obs, Hobs).flatten(1)
        g = torch.tensor(self.snrs, device=snr_db.device)
        key = g[(snr_db[:, None] - g[None]).abs().argmin(1)]
        out = torch.empty(Hobs.shape[0], self.W[self.snrs[0]][0].shape[0], dtype=Hobs.dtype, device=Hobs.device)
        for s in self.snrs:
            m = key == s
            if m.any():
                out[m] = half_cmatmul(y[m], *self.W[s])
        return out.view(Hobs.shape)


class Cascade:
    def __init__(self, stage1, prev_dirs, kmax2, speed_hi=800.0, device=os.environ.get('OTFS_DEV', 'cuda'), layout='std'):
        self.link = Link(layout=get_layout(layout), data_tx=(0,))
        self.link2 = wide_link(self.link, kmax2)
        self.anchor1 = AnchorBank(self.link, layout, speed_range=(30.0, speed_hi))
        self.nets = [load_net(stage1)]
        self.anchors = [None]
        for d in prev_dirs:
            self.nets.append(load_net(f'{d}/best.pt'))
            self.anchors.append(StageAnchors(d, None, self.link2))

    @torch.no_grad()
    def chain(self, d, upto=None):
        """Estimates of stages 1..upto (default all loaded stages)."""
        upto = len(self.nets) if upto is None else upto
        n, sc, _ = self.nets[0]
        H = regress(n, self.anchor1(d['Hls'], d['snr_db']), d['snr_db'], sc)
        out = [H]
        for k in range(1, upto):
            Hin = self.next_input(d, H)
            n, sc, _ = self.nets[k]
            H = regress(n, self.anchors[k](Hin, d['snr_db']), d['snr_db'], sc)
            out.append(H)
        return out

    def next_input(self, d, H):
        D = decide(self.link, d['Y'], H, d['guard'], d['snr_db'])
        return stage2_input(self.link, self.link2, d['Y'], H, D)


def train(a):
    dev = os.environ.get('OTFS_DEV', 'cuda')
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)
    prev = [x for x in a.prev.split(',') if x]
    cas = Cascade(a.stage1, prev, a.kmax2, a.speed_hi, layout=a.layout)
    tr = Scenario(PathBank(PATHS, TRAIN_CITIES, 'train', device=dev), cas.link, (30.0, a.speed_hi))
    va = Scenario(PathBank(PATHS, TRAIN_CITIES, 'val', device=dev), cas.link, (30.0, a.speed_hi))

    def producer_for_snr(s):
        def prod(bs, g):
            d = tr.draw(bs, g, snr=s)
            return d['H'], cas.next_input(d, cas.chain(d)[-1])
        return prod
    anc = StageAnchors(a.out, producer_for_snr, cas.link2)
    grid = torch.tensor(SNR_GRID, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)

    def draw(scen, B, g, snr=None):
        snr = grid[torch.randint(0, len(grid), (B,), generator=g, device=dev)] if snr is None else snr
        d = scen.draw(B, g, snr=snr)
        Hin = cas.next_input(d, cas.chain(d)[-1])
        d['A'] = anc(Hin, d['snr_db'])
        return d

    net = build_unet(a.model_channels, (1, 2, 4), (), num_res_blocks=2).to(dev)
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    fnet = torch.compile(net) if a.compile else net
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    ck = os.path.join(a.out, 'last.pt'); step, best, scale = 0, float('inf'), None
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s['net']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt'])
        sched.load_state_dict(s['sched']); step, best, scale = s['step'], s['best'], s['scale']
        gen.manual_seed(a.seed + step)
    if scale is None:
        with torch.no_grad():
            scale = math.sqrt(sum(to_real(d['H'] - d['A']).pow(2).mean().item()
                                  for d in (draw(tr, 64, gen) for _ in range(10))) / 10)
    cfg = dict(vars(a), mode='reg', anchor='cascade', scale=scale, channel_mult='1,2,4', attention='',
               num_res_blocks=2, params=sum(q.numel() for q in net.parameters()))
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    print(f"stage {len(prev) + 2}: params {cfg['params']/1e6:.2f} M, scale {scale:.3e}", flush=True)
    vgen = torch.Generator(device=dev)
    net.train(); t0 = time.time(); run = None
    while step < a.steps:
        with torch.no_grad():
            d = draw(tr, a.bs, gen)
        loss = regression_loss(fnet, d['H'], d['A'], d['snr_db'], scale)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0:
            print(f'step {step} loss {run:.4f} {(time.time()-t0)/500*1e3:.0f} ms/step', flush=True); t0 = time.time()
        if step % 5000 == 0 or step == a.steps:
            net.eval(); v = []
            for snr in (10.0, 20.0):
                vgen.manual_seed(12345); hs, gs = [], []
                for _ in range(4):
                    dv = draw(va, 64, vgen, snr=snr)
                    hs.append(regress(ema, dv['A'], dv['snr_db'], scale)); gs.append(dv['H'])
                v.append(nmse_db(torch.cat(hs), torch.cat(gs))[0])
            net.train()
            print(f'VAL step {step} NMSE@10/20dB {v[0]:.2f} {v[1]:.2f}', flush=True)
            score = sum(v) / 2
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=step, best=min(best, score), scale=scale), ck + '.tmp'); os.replace(ck + '.tmp', ck)
            if score < best:
                best = score
                torch.save(dict(ema=ema.state_dict(), scale=scale, step=step, val=v, config=cfg), os.path.join(a.out, 'best.pt'))
    print('done', flush=True)


@torch.no_grad()
def evaluate(a):
    dev = os.environ.get('OTFS_DEV', 'cuda')
    prev = [x for x in a.prev.split(',') if x]
    cas = Cascade(a.stage1, prev, a.kmax2, a.speed_hi, layout=a.layout)
    res = {}
    for split in a.splits.split(','):
        cities = TRAIN_CITIES if split == 'test' else UNSEEN_CITIES
        scen = Scenario(PathBank(PATHS, cities, 'test' if split == 'test' else 'all', device=dev), cas.link, (30.0, a.speed_hi))
        for snr in [float(x) for x in a.snrs.split(',')]:
            for v in [float(x) for x in a.speeds.split(',')]:
                errs = None; ber = None
                for b in range(a.n // 64):
                    g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(repr((split, snr, v, b)).encode()) % (2**31))
                    d = scen.draw(64, g, speed=v, snr=snr)
                    hs = cas.chain(d)
                    e = [nmse_db(h, d['H'])[1] for h in hs]
                    errs = e if errs is None else [torch.cat([x, y]) for x, y in zip(errs, e)]
                    bb = [detect_ber(cas.link, d['Y'], h, d['X'], d['bits'], d['guard'], d['snr_db'], streams=(0,)) for h in hs]
                    ber = [(x[0] + 0, x[1]) for x in bb] if ber is None else [(p[0] + q[0], p[1] + q[1]) for p, q in zip(ber, bb)]
                key = f'{split}_snr{snr:g}_v{v:g}'
                res[key] = dict(nmse_db=[10 * math.log10(x.mean().item()) for x in errs],
                                ber_stream0=[b[0] / b[1] for b in ber])
                print(key, [round(x, 2) for x in res[key]['nmse_db']], [f'{x:.2e}' for x in res[key]['ber_stream0']], flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'eval'])
    p.add_argument('--out', required=True)
    p.add_argument('--stage1', required=True)
    p.add_argument('--prev', default='')
    p.add_argument('--kmax2', type=int, default=14)
    p.add_argument('--layout', default='std')
    p.add_argument('--speed_hi', type=float, default=800.0)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--model_channels', type=int, default=32)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--snrs', default='10,20')
    p.add_argument('--speeds', default='250,500,650,800')
    p.add_argument('--splits', default='test,unseen')
    a = p.parse_args()
    train(a) if a.cmd == 'train' else evaluate(a)

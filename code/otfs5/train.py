"""Train the CRDM teacher (--mode diff) or the same-architecture regressor (--mode reg).

Channels are synthesised on the GPU from the path bank, so every step sees fresh
(user, speed, heading, SNR, data, noise) draws from the training cities.
"""
import argparse
import copy
import json
import math
import os
import time

import torch

from .diffusion import (alphas_for, build_unet, cosine_alphas_cumprod, ddim_sample, diffusion_loss,
                        net_call, regress, regression_loss, to_real)
from .anchor import SNR_GRID, make_anchor
from .layouts import get_layout
from .sim import TRAIN_CITIES, Link, PathBank, Scenario, nmse_db


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['diff', 'reg', 'kd'], required=True)
    p.add_argument('--teacher', default=None, help='checkpoint of the teacher (kd mode)')
    p.add_argument('--teacher_steps', default='1t:8', help="teacher sampler: 'reg' or e.g. '1t:8'")
    p.add_argument('--kd_alpha', type=float, default=1.0, help='weight of the teacher target')
    p.add_argument('--out', required=True)
    p.add_argument('--layout', default='std')
    p.add_argument('--steps', type=int, default=150_000)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--model_channels', type=int, default=64)
    p.add_argument('--channel_mult', default='1,2,4')
    p.add_argument('--attention', default='128,256')
    p.add_argument('--num_res_blocks', type=int, default=2)
    p.add_argument('--speed_lo', type=float, default=30.0)
    p.add_argument('--speed_hi', type=float, default=500.0)
    p.add_argument('--val_every', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--paths', default='__OTFS_ROOT__/paths')
    p.add_argument('--centre', action='store_true')
    p.add_argument('--single_stream', action='store_true', help='only antenna 0 carries data')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--pred', default='v', choices=['v', 'eps'], help='diffusion target: velocity or noise')
    p.add_argument('--schedule', default='cosine', choices=['cosine', 'linear'])
    p.add_argument('--anchor', default='ls', help="ls | lmmse (joint) | rx | rx<keep> (per receive antenna, e.g. rx0.25)")
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get('OTFS_DEV', 'cuda')
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lay = get_layout(a.layout)
    link = Link(layout=lay, data_tx=(0,) if a.single_stream else None)
    train = Scenario(PathBank(a.paths, TRAIN_CITIES, 'train', device=dev), link, (a.speed_lo, a.speed_hi), centre=a.centre)
    val = Scenario(PathBank(a.paths, TRAIN_CITIES, 'val', device=dev), link, (a.speed_lo, a.speed_hi), centre=a.centre)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    anc = make_anchor(a.anchor, link, a.layout, centre=a.centre, speed_range=(a.speed_lo, a.speed_hi)) if a.anchor != 'ls' else None
    grid = torch.tensor(SNR_GRID, device=dev)

    def draw(scen, B, g, snr=None):
        if anc is not None and snr is None:          # anchored training uses the anchor's SNR grid
            snr = grid[torch.randint(0, len(grid), (B,), generator=g, device=dev)]
        d = scen.draw(B, g, snr=snr)
        d['A'] = anc(d['Hls'], d['snr_db']) if anc is not None else d['Hls']
        return d

    cm = tuple(int(x) for x in a.channel_mult.split(','))
    att = tuple(int(x) for x in a.attention.split(',')) if a.attention else ()
    net = build_unet(a.model_channels, cm, att, num_res_blocks=a.num_res_blocks).to(dev)
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    nparam = sum(q.numel() for q in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    ab = alphas_for(a.schedule).to(dev)
    fnet = torch.compile(net) if a.compile else net   # shares parameters with net
    teacher = None
    if a.mode == 'kd':
        from .evaluate import load_net
        teacher = load_net(a.teacher)

    def teacher_residual(d):
        tn, tsc, tcfg = teacher
        if tcfg['mode'] == 'reg':
            Ht = regress(tn, d['A'], d['snr_db'], tsc)
        else:
            S, _, K = a.teacher_steps.partition(':')
            sp = 'trailing' if S.endswith('t') else 'leading'
            S, K = int(S.rstrip('t')), int(K or 1)
            Ht = sum(ddim_sample(tn, d['A'], d['snr_db'], tsc, ab, S=S, gen=gen, spacing=sp) for _ in range(K)) / K
        return Ht

    ck = os.path.join(a.out, 'last.pt')
    step, best, scale = 0, float('inf'), None
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s['net']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt'])
        sched.load_state_dict(s['sched']); step, best, scale = s['step'], s['best'], s['scale']
        gen.manual_seed(a.seed + step)
        print(f'resumed at step {step}', flush=True)
    if scale is None:
        with torch.no_grad():
            acc = 0.0
            for _ in range(20):
                d = draw(train, a.bs, gen)
                acc += to_real(d['H'] - d['A']).pow(2).mean().item()
            scale = math.sqrt(acc / 20)
    cfg_mode = 'reg' if a.mode == 'kd' else a.mode
    json.dump(dict(vars(a), mode=cfg_mode, train_mode=a.mode, scale=scale, params=nparam, overhead=lay.overhead(64, 128)),
              open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    print(f'params {nparam/1e6:.2f} M, residual scale {scale:.4e}, overhead {lay.overhead(64,128):.4f}', flush=True)

    vgen = torch.Generator(device=dev)
    def validate():
        vgen.manual_seed(12345)
        out = []
        for snr in (0.0, 10.0, 20.0):
            hs, gs = [], []
            for _ in range(4):
                d = draw(val, 64, vgen, snr=snr)
                if a.mode == 'diff':
                    hs.append(ddim_sample(ema, d['A'], d['snr_db'], scale, ab, S=1, gen=vgen, spacing='trailing', pred=a.pred))
                else:
                    hs.append(regress(ema, d['A'], d['snr_db'], scale))
                gs.append(d['H'])
            out.append(nmse_db(torch.cat(hs), torch.cat(gs))[0])
        return out

    net.train()
    t0, run = time.time(), 0.0
    while step < a.steps:
        with torch.no_grad():
            d = draw(train, a.bs, gen)
        if a.mode == 'diff':
            loss = diffusion_loss(fnet, d['H'], d['A'], d['snr_db'], scale, ab, pred=a.pred)
        elif a.mode == 'reg':
            loss = regression_loss(fnet, d['H'], d['A'], d['snr_db'], scale)
        else:
            with torch.no_grad():
                Ht = teacher_residual(d)
            target = a.kd_alpha * Ht + (1 - a.kd_alpha) * d['H']
            loss = regression_loss(fnet, target, d['A'], d['snr_db'], scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(a.ema).add_(pn.detach(), alpha=1 - a.ema)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f'step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/500*1e3:.0f} ms/step', flush=True)
            t0 = time.time()
        if step % a.val_every == 0 or step == a.steps:
            net.eval(); v = validate(); net.train()
            score = sum(v) / len(v)
            print(f'VAL step {step} NMSE@0/10/20dB {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}', flush=True)
            state = dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                         sched=sched.state_dict(), step=step, best=min(best, score), scale=scale)
            torch.save(state, ck + '.tmp'); os.replace(ck + '.tmp', ck)
            if score < best:
                best = score
                torch.save(dict(ema=ema.state_dict(), scale=scale, step=step, val=v,
                                config=json.load(open(os.path.join(a.out, 'config.json')))),
                           os.path.join(a.out, 'best.pt'))
    print('done', flush=True)


if __name__ == '__main__':
    main()

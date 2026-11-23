"""LMMSE anchors on a fixed SNR grid, cached on disk per pilot layout."""
import os

import torch

from .baselines import LMMSE, window_vectors
from .sim import TRAIN_CITIES, PathBank, Scenario

SNR_GRID = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


class AnchorBank:
    def __init__(self, link, layout_name, snrs=SNR_GRID, root='__OTFS_ROOT__/lmmse_v22',
                 paths='__OTFS_ROOT__/paths', device=os.environ.get('OTFS_DEV', 'cuda'), n_draws=int(os.environ.get('OTFS_ANCHOR_DRAWS', 16384)), centre=False,
                 speed_range=(30.0, 500.0), ue_rot=True, bs_rot=1.0471975511965976):
        ss = link.data_tx is not None
        os.makedirs(root, exist_ok=True)
        self.link, self.snrs, self.W = link, tuple(float(s) for s in snrs), {}
        scen = None
        tag = (f'{layout_name}{"_c" if centre else ""}_v{speed_range[0]:g}-{speed_range[1]:g}_n{n_draws}'
               f'_ue{int(ue_rot)}_bs{bs_rot:.3f}' + ('_ss' if ss else ''))
        for s in self.snrs:
            f = f'{root}/{tag}_snr{s:+.1f}.pt'
            if os.path.exists(f):
                W = torch.load(f, map_location='cpu')
            else:
                if scen is None:
                    scen = Scenario(PathBank(paths, TRAIN_CITIES, 'train', device=device), link,
                                    tuple(speed_range), centre=centre, ue_rot=ue_rot, bs_rot=bs_rot)
                W = LMMSE(scen, s, n_draws=n_draws).W
                torch.save(W, f + '.tmp'); os.replace(f + '.tmp', f)
                W = W.cpu(); torch.cuda.empty_cache()
            # half-precision real/imag parts on the GPU: 2 x 0.25 GB instead of 1 GB per SNR
            self.W[s] = (W.real.to(device, torch.float16), W.imag.to(device, torch.float16))
            del W

    def nearest(self, snr_db):
        g = torch.tensor(self.snrs, device=snr_db.device)
        return g[(snr_db[:, None] - g[None]).abs().argmin(1)]

    def __call__(self, Hls, snr_db):
        y = window_vectors(self.link, Hls).flatten(1)
        out = torch.empty(Hls.shape[0], self.W[self.snrs[0]][0].shape[0], dtype=Hls.dtype, device=Hls.device)
        key = self.nearest(snr_db)
        for s in self.snrs:
            m = key == s
            if m.any():
                out[m] = half_cmatmul(y[m], *self.W[s])
        return out.view(Hls.shape)


def half_cmatmul(y, Wr, Wi):
    """y [B,D] complex @ (Wr + j Wi)^T with fp16 weights and fp32 accumulation."""
    s = y.abs().amax().clamp_min(1e-30)                 # keep fp16 inputs in range
    yr, yi = (y.real / s).half(), (y.imag / s).half()
    re = (yr @ Wr.T).float() - (yi @ Wi.T).float()
    im = (yr @ Wi.T).float() + (yi @ Wr.T).float()
    return torch.complex(re, im) * s


class RxAnchorBank:
    """Per-receive-antenna LMMSE anchor (P2): the nt windows of one receive antenna (nt*W values) are mapped to
    the nt link grids of that antenna, with statistics pooled over the receive antennas.  Optionally only the
    fraction `keep` of output bins with the largest mean channel power is estimated (the rest is zero).
    Weights per SNR: keep * nt*N*M * nt*W complex, against nr*nt*N*M * nr*nt*W for the joint AnchorBank."""

    def __init__(self, link, layout_name, keep=1.0, snrs=SNR_GRID, root='__OTFS_ROOT__/lmmse_v22',
                 paths='__OTFS_ROOT__/paths', device=os.environ.get('OTFS_DEV', 'cuda'), n_draws=int(os.environ.get('OTFS_ANCHOR_DRAWS', 16384)), speed_range=(30.0, 500.0), ridge=1e-6):
        ss = link.data_tx is not None
        tag = f'rx_{layout_name}_v{speed_range[0]:g}-{speed_range[1]:g}_n{n_draws}' + ('_ss' if ss else '')
        self.link, self.snrs, self.W, self.keep = link, tuple(float(s) for s in snrs), {}, None
        scen = None
        for s in self.snrs:
            f = f'{root}/{tag}_snr{s:+.1f}.pt'
            if not os.path.exists(f):
                if scen is None:
                    scen = Scenario(PathBank(paths, TRAIN_CITIES, 'train', device=device), link, tuple(speed_range))
                g = torch.Generator(device=device); g.manual_seed(7)
                Ryy = Rhy = 0; P = 0; n = 0
                for _ in range(n_draws // 64):
                    d = scen.draw(64, g, snr=s)
                    y = window_vectors(link, d['Hls'])
                    yt = y.reshape(-1, y.shape[2] * y.shape[3])
                    ht = d['H'].reshape(-1, y.shape[2] * d['H'].shape[-2] * d['H'].shape[-1])
                    Ryy = Ryy + (yt.T @ yt.conj()).to(torch.complex128); Rhy = Rhy + ht.T @ yt.conj()
                    P = P + ht.abs().pow(2).sum(0); n += yt.shape[0]
                Ryy, Rhy = Ryy / n, Rhy / n
                Ryy = Ryy + ridge * torch.trace(Ryy).real / Ryy.shape[0] * torch.eye(Ryy.shape[0], device=device, dtype=Ryy.dtype)
                W = torch.linalg.solve(Ryy.T, Rhy.to(torch.complex128).T).T.to(torch.complex64)
                torch.save(dict(W=W.cpu(), P=P.cpu()), f + '.tmp'); os.replace(f + '.tmp', f)
                del Ryy, Rhy; torch.cuda.empty_cache()
            z = torch.load(f, map_location='cpu')
            idx = z['P'].argsort(descending=True)[:int(round(keep * z['P'].numel()))].sort().values
            self.idx = idx.to(device); self.n_out = z['P'].numel()   # same support for every SNR (power is SNR-free)
            W = z['W'][idx]
            self.W[s] = (W.real.to(device, torch.float16), W.imag.to(device, torch.float16))

    def weights(self):
        return self.W[self.snrs[0]][0].numel()

    def __call__(self, Hls, snr_db):
        y = window_vectors(self.link, Hls)
        B, nr = y.shape[:2]
        y = y.reshape(B * nr, -1)
        key = AnchorBank.nearest(self, snr_db).repeat_interleave(nr)
        out = torch.zeros(B * nr, self.n_out, dtype=Hls.dtype, device=Hls.device)
        for s in self.snrs:
            m = key == s
            if m.any():
                out[m.nonzero()[:, :1], self.idx[None]] = half_cmatmul(y[m], *self.W[s])
        return out.view(Hls.shape)


def make_anchor(kind, link, layout_name, **kw):
    """'lmmse' -> joint AnchorBank; 'rx' / 'rx<keep>' (e.g. rx0.25) -> RxAnchorBank."""
    if kind == 'lmmse':
        return AnchorBank(link, layout_name, **kw)
    if kind.startswith('rx'):
        kw.pop('centre', None)
        return RxAnchorBank(link, layout_name, keep=float(kind[2:] or 1.0), **kw)
    raise ValueError(kind)

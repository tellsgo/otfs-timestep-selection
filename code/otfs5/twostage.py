"""Data-aided two-stage estimation: estimate, decide data, cancel data leakage, re-read wider windows.

Single data stream from antenna 0 (the others send only their pilots), so the decisions come
from a per-bin SIMO LMMSE combiner without any genie knowledge.
"""
import os
import copy
import math

import torch

from .baselines import window_vectors
from .sim import Link


def pilot_frame(link: Link, B, device):
    N, M = link.num.N, link.num.M
    P = torch.zeros(B, link.nt, N, M, dtype=torch.complex64, device=device)
    for t in range(link.nt):
        P[:, t, link.layout.kp % N, (link.layout.lp + t * link.layout.dl) % M] = math.sqrt(link.layout.Ep)
    return P


def conv(Hrt, Xt):
    """sum_t H_rt (*) X_t for H [B,r,t,N,M], X [B,t,N,M] -> [B,r,N,M]."""
    return torch.fft.ifft2(torch.einsum('brtkl,btkl->brkl', torch.fft.fft2(Hrt), torch.fft.fft2(Xt)))


@torch.no_grad()
def decide(link: Link, Y, Hhat, guard, snr_db, t=0):
    """Hard QPSK decisions of stream t (the only data stream) by per-bin SIMO LMMSE."""
    N, M = link.num.N, link.num.M
    B = Y.shape[0]
    Z = torch.fft.fft2(Y - conv(Hhat, pilot_frame(link, B, Y.device)))
    h = torch.fft.fft2(Hhat[:, :, t])
    K = float((~guard).sum())
    q = (10.0 ** (-snr_db / 10.0) * N * M)[:, None, None]
    xf = (h.conj() * Z).sum(1) / (h.abs().pow(2).sum(1) + q / K)
    xh = torch.fft.ifft2(xf)
    d = (torch.sign(xh.real) + 1j * torch.sign(xh.imag)) / math.sqrt(2)
    d = torch.where(guard, torch.zeros_like(d), d)
    return d.to(torch.complex64)


def wide_link(link: Link, kmax2):
    """Same frame, wider read-out windows (only the receiver changes)."""
    l2 = copy.deepcopy(link)
    l2.layout = copy.deepcopy(link.layout)
    l2.layout.kmax = kmax2
    return l2


def stage2_input(link: Link, link2: Link, Y, H1, D, t=0):
    """Cancel the data of stream t with the stage-1 channel and read the wider windows."""
    Dfull = torch.zeros(Y.shape[0], link.nt, *Y.shape[-2:], dtype=Y.dtype, device=Y.device)
    Dfull[:, t] = D
    Y2 = Y - conv(H1, Dfull)
    return link2.ls(Y2)


class GenericLMMSE:
    """LMMSE from (H, observation tensor) pairs produced by a callable, windows of link_obs."""

    def __init__(self, producer, link_obs, n_draws=8192, bs=64, ridge=1e-6, seed=7, device=os.environ.get('OTFS_DEV', 'cuda')):
        g = torch.Generator(device=device); g.manual_seed(seed)
        Ryy = Rhy = None
        n = 0
        for _ in range(n_draws // bs):
            H, Hobs = producer(bs, g)
            y = window_vectors(link_obs, Hobs).flatten(1); h = H.flatten(1)
            if Ryy is None:
                Ryy = torch.zeros(y.shape[1], y.shape[1], dtype=torch.complex128, device=device)
                Rhy = torch.zeros(h.shape[1], y.shape[1], dtype=torch.complex64, device=device)
            Ryy += (y.T @ y.conj()).to(torch.complex128); Rhy += h.T @ y.conj(); n += bs
        Ryy /= n; Rhy /= n
        Ryy += ridge * torch.trace(Ryy).real / Ryy.shape[0] * torch.eye(Ryy.shape[0], device=device)
        # solve on the host: the full-grid cross-covariance can exceed the free GPU memory
        W = torch.linalg.solve(Ryy.cpu().T, Rhy.cpu().to(torch.complex128).T).T
        del Rhy, Ryy
        self.W = W.to(torch.complex64).to(device)
        self.link_obs, self.shape = link_obs, H.shape[1:]

    def __call__(self, Hobs):
        return (window_vectors(self.link_obs, Hobs).flatten(1) @ self.W.T).view(-1, *self.shape)

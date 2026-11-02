"""Classical estimators and the detector used for BER."""
import os
import math

import torch

from .sim import Link, Scenario


def window_vectors(link: Link, Hls):
    """Gather every antenna's window into y [B, nr, nt, W] (W = (2kmax+1)(lmax+1))."""
    N, M = link.num.N, link.num.M
    ys = []
    for t, (_, _, kc, lc) in enumerate(link.layout.window_index(N, M, Hls.device)):
        ys.append(Hls[:, :, t][:, :, kc[:, None], lc[None, :]].flatten(-2))
    return torch.stack(ys, 2)


class LMMSE:
    """Linear MMSE with empirical second-order statistics of the training cities.

    The window observations of all nr*nt links are stacked (MIMO-joint) and the
    whole DD grid is estimated, so the estimator also extrapolates leakage that
    falls outside the windows.  Statistics are collected per SNR and include the
    pilot and data interference exactly as the receiver sees it.
    """

    def __init__(self, scen: Scenario, snr_db, n_draws=16384, bs=64, seed=7, ridge=1e-6):
        dev = scen.bank.device
        g = torch.Generator(device=dev); g.manual_seed(seed)
        Ryy = Rhy = None
        n = 0
        for _ in range(n_draws // bs):
            d = scen.draw(bs, g, snr=snr_db)
            y = window_vectors(scen.link, d['Hls']).flatten(1)          # [B, D]
            h = d['H'].flatten(1)                                        # [B, F]
            if Ryy is None:
                Ryy = torch.zeros(y.shape[1], y.shape[1], dtype=torch.complex128, device=dev)
                Rhy = torch.zeros(h.shape[1], y.shape[1], dtype=torch.complex64, device=dev)
            Ryy += (y.T @ y.conj()).to(torch.complex128)
            Rhy += h.T @ y.conj()
            n += bs
        Ryy /= n; Rhy /= n
        Ryy += ridge * torch.trace(Ryy).real / Ryy.shape[0] * torch.eye(Ryy.shape[0], device=dev)
        # W = Rhy Ryy^{-1}, solved rather than inverted: W^T = Ryy^{-T} Rhy^T
        self.W = torch.linalg.solve(Ryy.T, Rhy.to(torch.complex128).T).T.to(torch.complex64)   # [F, D]
        self.link = scen.link
        self.shape = d['H'].shape[1:]

    def __call__(self, Hls):
        y = window_vectors(self.link, Hls).flatten(1)
        return (y @ self.W.T).view(-1, *self.shape)


class SOMP:
    """Simultaneous OMP over a fractional delay-Doppler dictionary.

    All nr*nt links share the path delays and Dopplers, so one support is found
    jointly; per-link gains are re-fitted by least squares after every pick.
    The selected atoms are then evaluated on the full grid.
    """

    def __init__(self, link: Link, k_step=0.25, l_step=0.25, k_pad=1.0, l_lo=-0.5, max_paths=10, thr=1.5):
        N, M = link.num.N, link.num.M
        lay = link.layout
        dev = os.environ.get('OTFS_DEV', 'cuda')
        kap = torch.arange(-lay.kmax - k_pad, lay.kmax + k_pad + 1e-9, k_step, device=dev)
        ell = torch.arange(l_lo, lay.lmax + 1e-9, l_step, device=dev)
        KK, LL = torch.meshgrid(kap, ell, indexing='ij')
        self.kap, self.ell = KK.flatten(), LL.flatten()
        kidx = torch.arange(N, device=dev, dtype=torch.float32)
        lidx = torch.arange(M, device=dev, dtype=torch.float32)
        Dk = torch.fft.fft(torch.exp(2j * math.pi * kidx * self.kap[:, None] / N), dim=-1) / N   # [A,N]
        Dl = torch.fft.ifft(torch.exp(-2j * math.pi * lidx * self.ell[:, None] / M), dim=-1)     # [A,M]
        self.full = (Dk[:, :, None] * Dl[:, None, :]).to(torch.complex64)                      # [A,N,M]
        _, _, kc, lc = lay.window_index(N, M, dev)[0]
        self.win = self.full[:, kc[:, None], lc[None, :]].flatten(1)                            # [A,W]
        self.win_n = self.win / self.win.norm(dim=1, keepdim=True)
        self.link, self.max_paths, self.thr = link, max_paths, thr
        self.W = self.win.shape[1]

    @torch.no_grad()
    def __call__(self, Hls, snr_db):
        B, nr, nt = Hls.shape[:3]
        y = window_vectors(self.link, Hls).reshape(B, nr * nt, self.W)                          # [B,L,W]
        noise = (10.0 ** (-snr_db / 10.0) / self.link.layout.Ep)[:, None]                      # per-bin variance
        stop = self.thr * noise * self.W * nr * nt
        r = y.clone()
        sel = torch.zeros(B, 0, dtype=torch.long, device=y.device)
        active = torch.ones(B, dtype=torch.bool, device=y.device)
        coef = torch.zeros(B, nr * nt, 0, dtype=y.dtype, device=y.device)
        for _ in range(self.max_paths):
            score = (r @ self.win_n.conj().T).abs().pow(2).sum(1)                                  # [B,A]
            pick = score.argmax(1)
            sel = torch.cat([sel, pick[:, None]], 1)
            A = self.win[sel]                                                                      # [B,K,W]
            G = A.conj() @ A.transpose(1, 2)                                                       # [B,K,K]
            rhs = y @ A.conj().transpose(1, 2)                                                     # [B,L,K]
            G = G + 1e-6 * torch.eye(G.shape[-1], device=y.device)
            c = torch.linalg.solve(G, rhs.transpose(1, 2)).transpose(1, 2)                         # [B,L,K]
            new_r = y - c @ A
            done = (new_r.abs().pow(2).sum((1, 2)) < stop[:, 0]) | ~active
            # keep the previous fit for samples that had already stopped
            if coef.shape[-1] == c.shape[-1] - 1:
                c = torch.where(active[:, None, None], c, torch.cat([coef, torch.zeros_like(c[..., :1])], -1))
            coef = c
            r = torch.where(active[:, None, None], new_r, r)
            active = active & ~done
            if not active.any():
                break
        full = self.full[sel]                                                                      # [B,K,N,M]
        H = torch.einsum('blk,bknm->blnm', coef, full)
        return H.view(B, nr, nt, *H.shape[-2:])


class GSOMP:
    """Guard-region simultaneous OMP with off-grid refinement.

    Observes the whole guard box of every receive antenna and models each path
    atom as excited by all nt pilots, so inter-pilot leakage is part of the model
    rather than an error.  The path support (fractional delay, Doppler) is shared
    by every link; each pick on the coarse grid is refined on a local fine grid,
    and per-link gains are re-fitted jointly by least squares.
    """

    def __init__(self, link: Link, k_step=0.25, l_step=0.25, k_pad=1.0, l_lo=-1.0, l_hi_pad=1.0,
                 max_paths=8, thr=1.2, fine=0.025, min_gain=0.03, sep=0.5, ridge=1e-2, window_only=True, device=os.environ.get('OTFS_DEV', 'cuda')):
        N, M = link.num.N, link.num.M
        lay = link.layout
        dev = device
        self.N, self.M, self.dev, self.link = N, M, dev, link
        kap = torch.arange(-lay.kmax - k_pad, lay.kmax + k_pad + 1e-9, k_step, device=dev)
        ell = torch.arange(l_lo, lay.lmax + l_hi_pad + 1e-9, l_step, device=dev)
        KK, LL = torch.meshgrid(kap, ell, indexing='ij')
        self.kap, self.ell = KK.flatten(), LL.flatten()
        # data-free core of the guard: bins no data symbol can reach under the design spreads
        core = torch.zeros(N, M, dtype=torch.bool, device=dev)
        ks = (lay.kp + torch.arange(-(lay.gk - lay.kmax), lay.gk - lay.kmax + 1, device=dev)) % N
        ls = (torch.arange(lay.lp - lay.gl_pre + lay.lmax, lay.lp + (lay.nt - 1) * lay.dl + lay.gl_post + 1, device=dev)) % M
        core[ks[:, None], ls[None, :]] = True
        self.ok, self.ol = torch.nonzero(core, as_tuple=True)
        self.phi = self._phi(self.kap, self.ell)                                                 # [A,nt,O]
        self.phin = self.phi / self.phi.norm(dim=-1, keepdim=True)
        h = k_step / 2
        off = torch.arange(-h, h + 1e-9, fine, device=dev)
        dk, dl = torch.meshgrid(off, off * (l_step / k_step), indexing='ij')
        self.dk, self.dl = dk.flatten(), dl.flatten()
        self.max_paths, self.thr, self.min_gain, self.sep, self.ridge = max_paths, thr, min_gain, sep, ridge
        # off-grid atoms fitted to interference extrapolate badly, so by default the
        # sparse model is only used to denoise the observed windows
        self.wmask = link.window_mask(dev) if window_only else None

    def _dk(self, kap):      # [...,N]
        n = torch.arange(self.N, device=self.dev, dtype=torch.float32)
        return torch.fft.fft(torch.exp(2j * math.pi * n * kap[..., None] / self.N), dim=-1) / self.N

    def _dl(self, ell):      # [...,M]
        m = torch.arange(self.M, device=self.dev, dtype=torch.float32)
        return torch.fft.ifft(torch.exp(-2j * math.pi * m * ell[..., None] / self.M), dim=-1)

    def _phi(self, kap, ell):
        """Responses of atoms (kap, ell) to every pilot, on the guard bins: [..., nt, O]."""
        lay = self.link.layout
        Dk, Dl = self._dk(kap), self._dl(ell)
        out = []
        for t in range(lay.nt):
            kk = (self.ok - lay.kp) % self.N
            ll = (self.ol - lay.lp - t * lay.dl) % self.M
            out.append((Dk[..., kk] * Dl[..., ll]).to(torch.complex64))
        return torch.stack(out, -2)

    @torch.no_grad()
    def __call__(self, Y, snr_db):
        lay = self.link.layout
        B, nr = Y.shape[:2]
        nt = lay.nt
        y = Y[:, :, self.ok, self.ol] / math.sqrt(lay.Ep)                                       # [B,nr,O]
        O = y.shape[-1]
        stop = self.thr * (10.0 ** (-snr_db / 10.0) / lay.Ep) * O * nr
        r = y.clone()
        kaps = torch.zeros(B, 0, device=y.device); ells = torch.zeros(B, 0, device=y.device)
        A = torch.zeros(B, 0, O, dtype=y.dtype, device=y.device)
        coef = None
        active = torch.ones(B, dtype=torch.bool, device=y.device)
        for _ in range(self.max_paths):
            score = torch.einsum('ato,bro->bart', self.phin.conj(), r).abs().pow(2).sum((-1, -2))   # [B,A]
            if kaps.shape[1]:   # minimum separation of half a bin from already selected paths
                near = ((self.kap[None, None] - kaps[:, :, None]).abs() < self.sep) & \
                       ((self.ell[None, None] - ells[:, :, None]).abs() < self.sep)
                score = score.masked_fill(near.any(1), -1.0)
            pick = score.argmax(1)
            ck = self.kap[pick][:, None] + self.dk[None]                                          # [B,C]
            cl = self.ell[pick][:, None] + self.dl[None]
            ph = self._phi(ck, cl)                                                                 # [B,C,nt,O]
            ph = ph / ph.norm(dim=-1, keepdim=True)
            sc2 = torch.einsum('bcto,bro->bcrt', ph.conj(), r).abs().pow(2).sum((-1, -2))         # [B,C]
            best = sc2.argmax(1)
            kb = ck.gather(1, best[:, None]); lb = cl.gather(1, best[:, None])
            kaps = torch.cat([kaps, kb], 1); ells = torch.cat([ells, lb], 1)
            A = torch.cat([A, self._phi(kb[:, 0], lb[:, 0])], 1)                                  # [B,K*nt,O]
            G = A.conj() @ A.transpose(1, 2)
            G = G + self.ridge * G.diagonal(dim1=1, dim2=2).real.mean(-1)[:, None, None] * torch.eye(A.shape[1], device=y.device)
            rhs = y @ A.conj().transpose(1, 2)                                                    # [B,nr,K*nt]
            c = torch.linalg.solve(G, rhs.transpose(1, 2)).transpose(1, 2)
            new_r = y - c @ A
            e_old, e_new = r.abs().pow(2).sum((1, 2)), new_r.abs().pow(2).sum((1, 2))
            # an atom that no longer explains >= min_gain of the residual is rejected
            accept = active & (e_new <= (1 - self.min_gain) * e_old)
            if coef is not None:
                prev = torch.cat([coef, torch.zeros_like(c[..., :nt])], -1)
                c = torch.where(accept[:, None, None], c, prev)
                new_r = torch.where(accept[:, None, None], new_r, r)
            else:
                c = torch.where(accept[:, None, None], c, torch.zeros_like(c))
                new_r = torch.where(accept[:, None, None], new_r, r)
            coef, r = c, new_r
            active = accept & (r.abs().pow(2).sum((1, 2)) >= stop)
            if not active.any():
                break
        K = kaps.shape[1]
        cc = coef.view(B, nr, K, nt)
        full = (self._dk(kaps)[..., :, None] * self._dl(ells)[..., None, :]).to(torch.complex64)  # [B,K,N,M]
        H = torch.einsum('brkt,bknm->brtnm', cc, full)
        return H * self.wmask[None, None] if self.wmask is not None else H


def threshold_estimator(link: Link, Hls, snr_db, c=3.0):
    """Raviteja et al. (TVT 2019): keep window bins whose magnitude exceeds c times the noise std."""
    sd = torch.sqrt(10.0 ** (-snr_db / 10.0) / link.layout.Ep)[:, None, None, None, None]
    return torch.where(Hls.abs() > c * sd, Hls, torch.zeros_like(Hls))


@torch.no_grad()
def detect_ber(link: Link, Y, Hhat, X, bits, guard, snr_db, Vf=None, streams=None):
    """Per-stream genie-aided SIC + receive LMMSE combining in the 2D-DFT domain, QPSK hard decision.

    For stream t the pilots and the other streams' data are cancelled with the
    *estimated* channels, so estimation errors on every link leave residual
    interference; the stream is then combined over the nr receive antennas.
    Vf [B,nr,nt,N,M], optional: second moment of the channel error in the 2D-DFT domain.  It
    enters as extra per-bin, per-antenna interference K * sum_{t in streams} Vf[r,t] + Ep * sum_t Vf[r,t]
    (data and pilots seen through the channel error); Vf = None is the usual mismatched receiver.
    """
    N, M = link.num.N, link.num.M
    P = torch.zeros_like(X)
    amp = math.sqrt(link.layout.Ep)
    for t in range(link.nt):
        P[:, t, link.layout.kp % N, (link.layout.lp + t * link.layout.dl) % M] = amp
    D = X - P                                                                     # data part only
    Hf = torch.fft.fft2(Hhat)                                                     # [B,r,t,N,M]
    Df = torch.fft.fft2(D)
    Yf = torch.fft.fft2(Y) - torch.einsum('brtkl,btkl->brkl', Hf, torch.fft.fft2(P))
    K = float((~guard).sum())
    errs = 0
    q = (10.0 ** (-snr_db / 10.0) * N * M)[:, None, None, None]                    # noise per bin
    streams = tuple(range(link.nt)) if streams is None else tuple(streams)
    if Vf is not None:
        # channel error times data of the data-carrying antennas, plus channel error times the pilots of all antennas
        q = q + K * Vf[:, :, list(streams)].sum(2) + link.layout.Ep * Vf.sum(2)      # [B,r,N,M]
    for t in streams:
        others = torch.einsum('brtkl,btkl->brkl', Hf, Df) - Hf[:, :, t] * Df[:, t][:, None]
        Z = Yf - others                                                           # [B,r,N,M]
        h = Hf[:, :, t]
        xf = (h.conj() * Z / q).sum(1) / ((h.abs().pow(2) / q).sum(1) + 1.0 / K)
        xh = torch.fft.ifft2(xf)
        bh = torch.stack([(xh.real < 0), (xh.imag < 0)], -1).long()
        errs += ((bh != bits[:, t]) & (~guard)[None, :, :, None]).sum().item()
    return errs, int((~guard).sum().item()) * Y.shape[0] * len(streams) * 2


def detect_ber_joint(link: Link, Y, Hhat, X, bits, guard, snr_db, Vf=None, streams=None):
    """Realizable joint MIMO detection: per-bin LMMSE over all data streams in the 2D-DFT domain, no genie.

    The pilots are cancelled with the estimated channel; at every bin f the data vector D[f] of the
    streams is estimated as (G^H Q^-1 G + I/K)^-1 G^H Q^-1 Y[f], with G = Hhat[f] (nr x ns), K the number
    of data bins (per-bin data power), Q = diag(MN sigma^2 + K sum_t Vf[r,t,f]) the per-antenna disturbance.
    Vf = None trusts the estimate.  QPSK hard decisions after the inverse 2D-DFT.
    """
    N, M = link.num.N, link.num.M
    P = torch.zeros_like(X)
    amp = math.sqrt(link.layout.Ep)
    for t in range(link.nt):
        P[:, t, link.layout.kp % N, (link.layout.lp + t * link.layout.dl) % M] = amp
    streams = tuple(range(link.nt)) if streams is None else tuple(streams)
    Hf = torch.fft.fft2(Hhat)                                                     # [B,r,t,N,M]
    Yf = torch.fft.fft2(Y) - torch.einsum('brtkl,btkl->brkl', Hf, torch.fft.fft2(P))
    K = float((~guard).sum())
    q = (10.0 ** (-snr_db / 10.0) * N * M)[:, None, None, None].expand(-1, Y.shape[1], N, M)
    if Vf is not None:
        q = q + K * Vf.sum(2)                                                     # [B,r,N,M]
    G = Hf[:, :, list(streams)].permute(0, 3, 4, 1, 2)                            # [B,N,M,r,s]
    y = Yf.permute(0, 2, 3, 1)[..., None]                                         # [B,N,M,r,1]
    qi = (1.0 / q).permute(0, 2, 3, 1)[..., None]                                 # [B,N,M,r,1]
    GhQ = (G * qi).conj().transpose(-1, -2)                                       # G^H Q^-1
    A = GhQ @ G + torch.eye(len(streams), dtype=G.dtype, device=G.device) / K
    xf = torch.linalg.solve(A, GhQ @ y)[..., 0].permute(0, 3, 1, 2)               # [B,s,N,M]
    xh = torch.fft.ifft2(xf)
    bh = torch.stack([(xh.real < 0), (xh.imag < 0)], -1).long()                   # [B,s,N,M,2]
    errs = ((bh != bits[:, list(streams)]) & (~guard)[None, None, :, :, None]).sum().item()
    return errs, int((~guard).sum().item()) * Y.shape[0] * len(streams) * 2


def detect_ber_cov(link: Link, Y, Hhat, X, bits, guard, snr_db, Cf, Vf, t=0):
    """Single-stream detection with the full receive-antenna covariance of the channel error per bin.
    Cf [N,M,nr,nr]: E[Delta_{r t} Delta_{r' t}^*] of the data antenna t; Vf [nr,nt,N,M] for the pilot term (diagonal)."""
    N, M = link.num.N, link.num.M
    P = torch.zeros_like(X)
    amp = math.sqrt(link.layout.Ep)
    for u in range(link.nt):
        P[:, u, link.layout.kp % N, (link.layout.lp + u * link.layout.dl) % M] = amp
    Hf = torch.fft.fft2(Hhat)
    Z = (torch.fft.fft2(Y) - torch.einsum('brtkl,btkl->brkl', Hf, torch.fft.fft2(P))).permute(0, 2, 3, 1)[..., None]  # [B,N,M,r,1]
    K = float((~guard).sum()); nr = Y.shape[1]
    noise = (10.0 ** (-snr_db / 10.0) * N * M)[:, None, None, None, None]
    pil = (link.layout.Ep * Vf.sum(1)).permute(1, 2, 0)                          # [N,M,r]
    Q = noise * torch.eye(nr, device=Y.device, dtype=Z.dtype) + K * Cf[None] + torch.diag_embed(pil)[None].to(Z.dtype)
    h = Hf[:, :, t].permute(0, 2, 3, 1)[..., None]                                # [B,N,M,r,1]
    Qi_h = torch.linalg.solve(Q, h)
    xf = ((Qi_h.conj().transpose(-1, -2) @ Z)[..., 0, 0]) / ((h.conj().transpose(-1, -2) @ Qi_h)[..., 0, 0].real + 1.0 / K)
    xh = torch.fft.ifft2(xf)
    bh = torch.stack([(xh.real < 0), (xh.imag < 0)], -1).long()
    errs = ((bh != bits[:, t]) & (~guard)[None, :, :, None]).sum().item()
    return errs, int((~guard).sum().item()) * Y.shape[0] * 2

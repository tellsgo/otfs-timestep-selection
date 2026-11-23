"""MIMO-OTFS link simulator with path-level doubly-dispersive channels (data v2).

Grid convention (kept from the CRDM code base): tensors are [..., N, M] with
N = 64 Doppler bins (index k, OTFS symbols) and M = 128 delay bins (index l,
subcarriers). The DD input-output relation is the ideal-pulse 2D circular
convolution  y_r[k,l] = sum_t sum_{k',l'} h_rt[k',l'] x_t[k-k', l-l'] + w_r[k,l].

Each ray-traced path p contributes  g_p a_r(p) a_t(p)^* D_N(k - kappa_p) D_M(l - ell_p)
with kappa_p = nu_p / dnu and ell_p = tau_p / dtau fractional, so every path has
its own Doppler  nu_p = (v / lambda) sin(theta_A,p) cos(phi_A,p - psi).
"""
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

C0 = 299_792_458.0
TRAIN_CITIES = ('city_1_losangeles', 'city_2_chicago', 'city_3_houston', 'city_4_phoenix',
                'city_7_sandiego', 'city_8_dallas', 'city_12_fortworth', 'city_18_denver')
UNSEEN_CITIES = ('city_0_newyork',)


@dataclass
class Numerology:
    fc: float = 3.5e9          # carrier (Hz), DeepMIMO city ray tracing band
    df: float = 15e3           # subcarrier spacing (Hz)
    M: int = 128               # delay bins (subcarriers)
    N: int = 64                # Doppler bins (symbols)

    @property
    def T(self): return 1.0 / self.df
    @property
    def Tf(self): return self.N * self.T
    @property
    def B(self): return self.M * self.df
    @property
    def dtau(self): return 1.0 / self.B
    @property
    def dnu(self): return 1.0 / self.Tf
    @property
    def lam(self): return C0 / self.fc


class PathBank:
    """Ray-traced path parameters of one or more cities, split by user."""

    def __init__(self, root, cities, split='train', seed=0, device=os.environ.get('OTFS_DEV', 'cuda')):
        keys = ('power', 'phase', 'toa', 'aoa_phi', 'aoa_theta', 'aod_phi', 'aod_theta')
        cols = {k: [] for k in keys}
        for c in cities:
            z = np.load(f'{root}/{c}.npz')
            n = z['npath'].shape[0]
            perm = np.random.default_rng(seed).permutation(n)
            if c in UNSEEN_CITIES or split == 'all':
                sel = perm
            else:
                a, b = int(0.8 * n), int(0.9 * n)
                sel = {'train': perm[:a], 'val': perm[a:b], 'test': perm[b:]}[split]
            for k in keys:
                cols[k].append(z[k][sel])
        cat = {k: np.concatenate(v, 0) for k, v in cols.items()}
        pw = cat['power']
        valid = np.isfinite(pw)
        amp = np.where(valid, np.sqrt(np.where(valid, pw, 0.0)), 0.0)   # 'power' is linear watts
        ph = np.deg2rad(cat['phase'])
        toa = np.where(valid, cat['toa'], np.inf)
        tau = np.where(valid, toa - toa.min(1, keepdims=True), 0.0)  # timing aligned to first path
        d = lambda x: torch.as_tensor(x, device=device)
        clean = lambda x: np.where(valid, np.nan_to_num(np.deg2rad(x)), 0.0)    # padded paths carry no angle
        self.g = d(np.where(valid, amp * np.exp(1j * np.nan_to_num(ph)), 0.0)).to(torch.complex64)
        self.tau = d(tau).float()
        self.aoa_phi, self.aoa_theta = d(clean(cat['aoa_phi'])).float(), d(clean(cat['aoa_theta'])).float()
        self.aod_phi, self.aod_theta = d(clean(cat['aod_phi'])).float(), d(clean(cat['aod_theta'])).float()
        self.n_users, self.P = self.g.shape
        self.device = device


def _dirichlet_fft(frac, n_bins, inverse):
    """D[b,p,i] = (1/n) sum_n exp(+-j 2 pi n (frac - i)/n) evaluated on i = 0..n-1."""
    n = torch.arange(n_bins, device=frac.device, dtype=torch.float32)
    if inverse:   # delay axis: (1/M) sum_m exp(j2pi m (l - ell)/M)
        e = torch.exp(-2j * math.pi * n * frac[..., None] / n_bins)
        return torch.fft.ifft(e, dim=-1)
    e = torch.exp(2j * math.pi * n * frac[..., None] / n_bins)   # Doppler axis
    return torch.fft.fft(e, dim=-1) / n_bins


def ula(nant, theta, phi):
    """Half-wavelength ULA along x: a[n] = exp(j pi n sin(theta) cos(phi))."""
    n = torch.arange(nant, device=theta.device, dtype=torch.float32)
    return torch.exp(1j * math.pi * n * (torch.sin(theta) * torch.cos(phi))[..., None])


def synth_dd_channel(bank: PathBank, idx, speed_kmh, heading, num: Numerology, nr=4, nt=4,
                     ue_rot=None, bs_rot=None):
    """Return H [B, nr, nt, N, M] complex64, per-sample normalised to unit mean link energy.

    ue_rot, bs_rot [B] (rad): azimuth rotation of the UE / BS array axis (orientation diversity).
    """
    g = bank.g[idx]
    tau = bank.tau[idx]
    v = (speed_kmh / 3.6)[:, None]
    nu = v / num.lam * torch.sin(bank.aoa_theta[idx]) * torch.cos(bank.aoa_phi[idx] - heading[:, None])
    kappa = nu / num.dnu
    ell = tau / num.dtau
    Dn = _dirichlet_fft(kappa, num.N, inverse=False)                 # [B,P,N]
    Dm = _dirichlet_fft(ell, num.M, inverse=True)                    # [B,P,M]
    pa = bank.aoa_phi[idx] - (ue_rot[:, None] if ue_rot is not None else 0.0)
    pd = bank.aod_phi[idx] - (bs_rot[:, None] if bs_rot is not None else 0.0)
    ar = ula(nr, bank.aoa_theta[idx], pa)                            # [B,P,nr]
    at = ula(nt, bank.aod_theta[idx], pd)                            # [B,P,nt]
    coef = g[:, :, None, None] * ar[:, :, :, None] * at.conj()[:, :, None, :]   # [B,P,nr,nt]
    H = torch.einsum('bprt,bpk,bpl->brtkl', coef, Dn.to(torch.complex64), Dm.to(torch.complex64))
    e = H.abs().pow(2).sum((-1, -2)).mean((1, 2))
    return H / torch.sqrt(e.clamp_min(1e-30))[:, None, None, None, None]


@dataclass
class PilotLayout:
    """Embedded pilots of nt antennas placed along the delay axis inside one guard region.

    Antenna t sends one impulse at (kp, lp + t*dl).  Its observation window is
    k in kp + [-kmax, kmax], l in lp + t*dl + [0, lmax].  The guard is the set of
    bins whose data would reach some window (Raviteja et al., TVT 2019), clipped
    by the optional shrink factors used for reduced-overhead layouts:
      Doppler extent  kp + [-gk, gk]               (gk = 2 kmax by default, N//2 = full)
      delay extent    lp + [-gl_pre, (nt-1) dl + gl_post]
    """
    kmax: int = 7
    lmax: int = 3
    nt: int = 4
    kp: int = 32
    lp: int = 40
    dl: Optional[int] = None       # pilot spacing in delay; default lmax+1 (non-overlapping windows)
    gk: Optional[int] = None       # Doppler half-extent of the guard
    gl_pre: Optional[int] = None   # delay guard before the first pilot
    gl_post: Optional[int] = None  # delay guard after the last pilot
    pilot_db: float = 20.0         # pilot-to-data symbol energy ratio

    def __post_init__(self):
        if self.dl is None: self.dl = self.lmax + 1
        if self.gk is None: self.gk = 2 * self.kmax
        if self.gl_pre is None: self.gl_pre = self.lmax
        if self.gl_post is None: self.gl_post = self.lmax
        self.Ep = 10.0 ** (self.pilot_db / 10.0)

    def masks(self, N, M, device):
        guard = torch.zeros(N, M, dtype=torch.bool, device=device)
        ks = (self.kp + torch.arange(-self.gk, self.gk + 1, device=device)) % N
        l0 = self.lp - self.gl_pre
        l1 = self.lp + (self.nt - 1) * self.dl + self.gl_post
        ls = torch.arange(l0, l1 + 1, device=device) % M
        guard[ks[:, None], ls[None, :]] = True
        return guard

    def overhead(self, N, M):
        return float(self.masks(N, M, 'cpu').float().mean())

    def window_index(self, N, M, device):
        """(rx-grid k, rx-grid l, channel k, channel l) for each antenna's window."""
        dk = torch.arange(-self.kmax, self.kmax + 1, device=device)
        dlv = torch.arange(0, self.lmax + 1, device=device)
        out = []
        for t in range(self.nt):
            kr = (self.kp + dk) % N
            lr = (self.lp + t * self.dl + dlv) % M
            out.append((kr, lr, dk % N, dlv % M))
        return out


@dataclass
class Link:
    num: Numerology = field(default_factory=Numerology)
    layout: PilotLayout = field(default_factory=PilotLayout)
    nr: int = 4
    nt: int = 4
    data_tx: Optional[tuple] = None      # antennas that carry data (None = all); the others send only pilots

    def transmit(self, B, gen, device):
        N, M = self.num.N, self.num.M
        guard = self.layout.masks(N, M, device)
        bits = torch.randint(0, 2, (B, self.nt, N, M, 2), generator=gen, device=device)
        sym = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / math.sqrt(2)
        X = torch.where(guard, torch.zeros((), dtype=sym.dtype, device=device), sym).to(torch.complex64)
        if self.data_tx is not None:
            off = [t for t in range(self.nt) if t not in self.data_tx]
            X[:, off] = 0
        amp = math.sqrt(self.layout.Ep)
        for t in range(self.nt):
            X[:, t, self.layout.kp % N, (self.layout.lp + t * self.layout.dl) % M] = amp
        return X, bits, guard

    def receive(self, H, X, snr_db, gen):
        """Y [B,nr,N,M] = sum_t H_rt (*) X_t + noise, with Es = 1 per data symbol."""
        Yf = torch.einsum('brtkl,btkl->brkl', torch.fft.fft2(H), torch.fft.fft2(X))
        Y = torch.fft.ifft2(Yf)
        sig2 = 10.0 ** (-snr_db / 10.0)
        w = torch.randn(Y.shape, generator=gen, device=Y.device, dtype=torch.complex64)  # unit variance
        return Y + torch.sqrt(sig2)[:, None, None, None] * w

    def ls(self, Y):
        """Read every antenna's pilot window; returns H_ls [B,nr,nt,N,M] (zero outside windows)."""
        B, nr, N, M = Y.shape
        Hls = torch.zeros(B, nr, self.nt, N, M, dtype=Y.dtype, device=Y.device)
        amp = math.sqrt(self.layout.Ep)
        for t, (kr, lr, kc, lc) in enumerate(self.layout.window_index(N, M, Y.device)):
            Hls[:, :, t, kc[:, None], lc[None, :]] = Y[:, :, kr[:, None], lr[None, :]] / amp
        return Hls

    def window_mask(self, device):
        N, M = self.num.N, self.num.M
        m = torch.zeros(self.nt, N, M, dtype=torch.bool, device=device)
        for t, (_, _, kc, lc) in enumerate(self.layout.window_index(N, M, device)):
            m[t, kc[:, None], lc[None, :]] = True
        return m


@dataclass
class Scenario:
    """Draws (user, speed, heading, SNR) and returns everything an estimator needs."""
    bank: PathBank
    link: Link
    speed_range: tuple = (30.0, 500.0)
    snr_range: tuple = (-5.0, 25.0)
    centre: bool = False         # receiver-side Doppler centring on the acquired dominant path
    ue_rot: bool = True          # uniform random UE array azimuth (handheld orientation)
    bs_rot: float = math.pi / 3  # BS array azimuth drawn uniformly in [-bs_rot, bs_rot] (rad)

    def draw(self, B, gen, speed=None, snr=None):
        dev = self.bank.device
        idx = torch.randint(0, self.bank.n_users, (B,), generator=gen, device=dev)
        lo, hi = self.speed_range
        v = torch.full((B,), float(speed), device=dev) if speed is not None else lo + (hi - lo) * torch.rand(B, generator=gen, device=dev)
        psi = 2 * math.pi * torch.rand(B, generator=gen, device=dev)
        lo, hi = self.snr_range
        if torch.is_tensor(snr):
            s = snr.to(dev).float()
        elif snr is not None:
            s = torch.full((B,), float(snr), device=dev)
        else:
            s = lo + (hi - lo) * torch.rand(B, generator=gen, device=dev)
        ur = 2 * math.pi * torch.rand(B, generator=gen, device=dev) if self.ue_rot else None
        br = self.bs_rot * (2 * torch.rand(B, generator=gen, device=dev) - 1) if self.bs_rot > 0 else None
        H = synth_dd_channel(self.bank, idx, v, psi, self.link.num, self.link.nr, self.link.nt, ur, br)
        X, bits, guard = self.link.transmit(B, gen, dev)
        Y = self.link.receive(H, X, s, gen)
        k0 = torch.zeros(B, device=dev)
        if self.centre:
            k0 = acquire_doppler(Y, self.link.layout, self.link.num.N, self.link.num.M)
            Y, H = doppler_shift(Y, k0), doppler_shift(H, k0)   # unitary: NMSE and BER are unchanged
        Hls = self.link.ls(Y)
        return dict(H=H, Hls=Hls, Y=Y, X=X, bits=bits, guard=guard, snr_db=s, speed=v, idx=idx, kappa0=k0)


def nmse_db(Hhat, H):
    e = (Hhat - H).abs().pow(2).flatten(1).sum(1) / H.abs().pow(2).flatten(1).sum(1)
    return 10 * torch.log10(e.mean()).item(), e


def doppler_shift(Z, kappa0):
    """Fractional circular shift of a DD tensor along Doppler: peak at kappa -> kappa - kappa0.

    Implemented as a modulation in the symbol-time domain of the Doppler axis (axis -2), so
    applying it to Y equals applying it to every H_rt in (H (*) X): data, pilots and noise
    statistics are unchanged.  kappa0 has shape [B].
    """
    N = Z.shape[-2]
    i = torch.arange(N, device=Z.device, dtype=torch.float32)
    n = (-i) % N                                                    # FFT bin i along Doppler <-> symbol time n
    ramp = torch.exp(-2j * math.pi * n[None, :] * kappa0[:, None] / N).to(Z.dtype)   # [B,N]
    shape = [Z.shape[0]] + [1] * (Z.dim() - 3) + [N, 1]
    return torch.fft.ifft(torch.fft.fft(Z, dim=-2) * ramp.view(shape), dim=-2)


def acquire_doppler(Y, layout, N, M, span=None, fine=0.02):
    """Dominant Doppler of the received pilot responses (all antennas), integer + fractional.

    Searches the whole Doppler axis inside the pilot delay band; the pilots are 20 dB above a
    data symbol, so the dominant path dominates the band even when it lies outside the window.
    """
    lay = layout
    l0 = lay.lp
    l1 = lay.lp + (lay.nt - 1) * lay.dl + lay.lmax
    band = Y[:, :, :, l0:l1 + 1]                                    # [B,nr,N,L]
    prof = band.abs().pow(2).sum((1, 3))                           # [B,N]
    kint = prof.argmax(1)
    kint = torch.where(kint > N // 2, kint - N, kint) - lay.kp % N  # relative to the pilot row
    kint = (kint + N // 2) % N - N // 2
    offs = torch.arange(-1.0, 1.0 + 1e-9, fine, device=Y.device)
    cand = kint[:, None].float() + offs[None]                       # [B,C]
    n = torch.arange(N, device=Y.device, dtype=torch.float32)
    D = torch.fft.fft(torch.exp(2j * math.pi * n * cand[..., None] / N), dim=-1) / N   # [B,C,N]
    D = torch.roll(D, shifts=lay.kp % N, dims=-1)                   # response of a pilot at row kp
    D = D / D.norm(dim=-1, keepdim=True)
    sc = torch.einsum('bcn,brnl->bcrl', D.conj(), band.to(D.dtype)).abs().pow(2).sum((-1, -2))
    return cand.gather(1, sc.argmax(1, keepdim=True))[:, 0]

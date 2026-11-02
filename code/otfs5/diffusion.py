"""Conditional residual diffusion (CRDM) pieces shared by every experiment.

The network sees  c + x_t  where c = H_ls / s and x_t is the noised residual
R = (H - H_ls) / s, exactly as in the CRDM reference implementation; it predicts
v = sqrt(ab) eps - sqrt(1 - ab) R.  s is one global RMS fixed at training start.
"""
import math

import torch
import torch.nn.functional as F

from .unet import OTFSDiffusionModel

T_TRAIN = 1000
SNR_LO, SNR_HI = -5.0, 25.0


def cosine_alphas_cumprod(T=T_TRAIN, max_beta=0.999):
    """Same as diffusers 'squaredcos_cap_v2'."""
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = [min(1 - f((i + 1) / T) / f(i / T), max_beta) for i in range(T)]
    return torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), 0).float()


def build_unet(model_channels=64, channel_mult=(1, 2, 4), attention=(128, 256), dropout=0.1, num_res_blocks=2):
    return OTFSDiffusionModel(
        in_channels=8, model_channels=model_channels, channel_mult=channel_mult,
        num_res_blocks=num_res_blocks, attention_resolutions=attention, dropout=dropout,
        time_embed_dim=512, use_class_condition=True, use_explicit_snr=True,
        use_sinusoidal_snr_embed=True, use_complex_unet=True, use_conditioning=True,
        use_speed_cond=False, use_snr_cond=True)


def to_real(H):            # [B,4,4,N,M] complex -> [B,8,4,N,M]
    return torch.cat([H.real, H.imag], 1)


def to_complex(x):
    n = x.shape[1] // 2
    return torch.complex(x[:, :n].float(), x[:, n:].float())


def snr_code(snr_db):
    """Map SNR in dB to the [0, 10] range the sinusoidal SNR embedding expects."""
    return ((snr_db - SNR_LO) / (SNR_HI - SNR_LO) * 10.0).clamp(0.0, 10.0)


def net_call(net, x, t, snr_db):
    zeros = torch.zeros_like(snr_db)
    return net(x, t, speed_kmh=zeros, snr_db=snr_code(snr_db))


def linear_alphas_cumprod(T=T_TRAIN, b0=1e-4, b1=0.02):
    """DDPM linear beta schedule (Ho et al. 2020); its terminal signal power alpha_bar_{T-1} is about 4e-5, not zero."""
    return torch.cumprod(1 - torch.linspace(b0, b1, T, dtype=torch.float64), 0).float()


def alphas_for(schedule):
    return linear_alphas_cumprod() if schedule == 'linear' else cosine_alphas_cumprod()


def ddim_timesteps(S, T=T_TRAIN, spacing='leading'):
    """'leading' (diffusers default, used by CRDM): [(S-1)r, ..., r, 0], never visits t = T-1.
    'trailing' (Lin et al., WACV 2024): [T-1, T-1-r, ...], the first step starts from pure noise."""
    if spacing == 'trailing':
        return [int(round(T - i * T / S)) - 1 for i in range(S)]
    r = T // S
    return list(range((S - 1) * r, -1, -r))


@torch.no_grad()
def ddim_sample(net, Hls, snr_db, scale, ab, S=5, gen=None, x_T=None, return_traj=False, t_start=None,
                spacing='leading', pred='v'):
    """Deterministic DDIM (eta = 0) on the normalised residual.  Returns H_hat (complex)."""
    c = to_real(Hls) / scale
    x = torch.randn(c.shape, generator=gen, device=c.device) if x_T is None else x_T
    ts = ddim_timesteps(S, spacing=spacing) if t_start is None else t_start
    traj = []
    stride = T_TRAIN // len(ts)
    for i, t in enumerate(ts):
        a_t = ab[t]
        if t_start is None:     # diffusers convention: previous timestep is t - floor(T/S)
            tp = t - stride
            a_prev = ab[tp] if tp >= 0 else torch.tensor(1.0, device=c.device)
        else:
            a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=c.device)
        tt = torch.full((c.shape[0],), t, device=c.device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v = net_call(net, c + x, tt, snr_db).float()
        if pred == 'eps':                # the network predicts the noise
            eps = v
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        else:
            x0 = a_t.sqrt() * x - (1 - a_t).sqrt() * v
            eps = (1 - a_t).sqrt() * x + a_t.sqrt() * v
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        if return_traj:
            traj.append(Hls + to_complex(x0) * scale)
    Hhat = Hls + to_complex(x) * scale
    return (Hhat, traj) if return_traj else Hhat


@torch.no_grad()
def regress(net, Hls, snr_db, scale):
    c = to_real(Hls) / scale
    tt = torch.zeros(c.shape[0], device=c.device, dtype=torch.long)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        r = net_call(net, c, tt, snr_db).float()
    return Hls + to_complex(r) * scale


def sample_train_t(B, device, ddim_prob=0.3, grids=(5, 10, 20, 50, 100)):
    t = torch.randint(0, T_TRAIN, (B,), device=device)
    if torch.rand(()) < ddim_prob:
        S = grids[int(torch.randint(0, len(grids), ()))]
        g = torch.tensor(ddim_timesteps(S), device=device)
        t = g[torch.randint(0, len(g), (B,), device=device)]
    return t


def diffusion_loss(net, H, Hls, snr_db, scale, ab, pred='v'):
    R = to_real(H - Hls) / scale
    c = to_real(Hls) / scale
    t = sample_train_t(R.shape[0], R.device)
    a = ab[t].view(-1, 1, 1, 1, 1)
    eps = torch.randn_like(R)
    x_t = a.sqrt() * R + (1 - a).sqrt() * eps
    target = eps if pred == 'eps' else a.sqrt() * eps - (1 - a).sqrt() * R
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net_call(net, c + x_t, t, snr_db)
    return F.mse_loss(out.float(), target)


def regression_loss(net, H, Hls, snr_db, scale):
    R = to_real(H - Hls) / scale
    c = to_real(Hls) / scale
    t = torch.zeros(R.shape[0], device=R.device, dtype=torch.long)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = net_call(net, c, t, snr_db)
    return F.mse_loss(pred.float(), R)

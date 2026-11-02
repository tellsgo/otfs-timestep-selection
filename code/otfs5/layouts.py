"""Named pilot layouts.  'std' is the conventional per-antenna embedded-pilot design."""
from .sim import PilotLayout


def get_layout(name):
    if name == 'std':          # kmax covers 500 km/h, lmax covers the 99th-percentile excess delay
        return PilotLayout()
    if name == 'fullk':        # fractional-Doppler-safe guard spanning the whole Doppler axis
        return PilotLayout(gk=32)
    if name.startswith('w'):   # conventional guard of kmax=7 read over a wider Doppler window, e.g. w14
        K = int(name[1:])
        return PilotLayout(kmax=K, gk=14)
    if name.startswith('p1'):  # single-antenna embedded pilot (one transmit antenna only)
        return PilotLayout(nt=1)
    if name.startswith('dc'):  # Doppler-centred design: guard sized for a residual spread of K bins
        K = int(name[2:])
        return PilotLayout(kmax=K, gk=2 * K)
    if name.startswith('k'):   # conventional design for a larger maximum Doppler, e.g. k11 for 800 km/h
        K = int(name[1:])
        return PilotLayout(kmax=K)
    if name.startswith('L'):   # general: L<gk>_<dl>_<gpre>_<gpost>, e.g. L14_3_2_1
        gk, dl, gpre, gpost = (int(x) for x in name[1:].split('_'))
        return PilotLayout(gk=gk, dl=dl, gl_pre=gpre, gl_post=gpost)
    if name.startswith('c'):   # compressed: c<gk>_<dl>_<glpost>, e.g. c7_2_1
        gk, dl, gpost = (int(x) for x in name[1:].split('_'))
        return PilotLayout(gk=gk, dl=dl, gl_post=gpost, gl_pre=min(3, gpost + 2))
    raise ValueError(name)

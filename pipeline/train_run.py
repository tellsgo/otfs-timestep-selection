"""Train one network of this package from its saved configuration, after the networks it depends on.

  python pipeline/train_run.py NAME            # configs/runs/NAME.json -> runs_v21/NAME (skipped if DONE exists)
  SMOKE=1 python pipeline/train_run.py NAME    # 20 steps into smoke/runs_v21/NAME, to check that the environment works

The configuration is the config.json that the original training run wrote.  Teacher, first-stage and previous-stage
networks named in it are trained first.  Interrupted runs resume from last.pt.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get('PY', sys.executable)
SMOKE = os.environ.get('SMOKE') == '1'
RUNS = os.path.join(ROOT, 'smoke' if SMOKE else '', 'runs_v21')
OLD = 'runs_v21/'


def cfg_of(name):
    return json.load(open(os.path.join(ROOT, 'configs', 'runs', name + '.json')))


def dep(path):
    """runs_v21/<name>/best.pt or runs_v21/<name> in a stored config -> (name, path in this package)."""
    rel = path[len(OLD):] if path.startswith(OLD) else None
    if rel is None:
        raise SystemExit(f'unexpected path in config: {path}')
    name = rel.split('/')[0]
    return name, os.path.join(RUNS, rel)


def argv_of(name, c):
    out = os.path.join(RUNS, name)
    steps = 20 if SMOKE else c['steps']
    deps, args = [], []
    if c.get('anchor') == 'cascade':                      # otfs5.cascade train
        s1, s1p = dep(c['stage1']); deps.append(s1)
        args = ['-m', 'otfs5.cascade', 'train', '--out', out, '--stage1', s1p]
        if c.get('prev'):
            p, pp = dep(c['prev']); deps.append(p); args += ['--prev', pp]
        for k in ('kmax2', 'layout', 'speed_hi', 'bs', 'lr', 'model_channels', 'seed'):
            if k in c:                                    # older configs lack some keys: the defaults apply, as in the original run
                args += [f'--{k}', str(c[k])]
        args += ['--steps', str(steps)] + (['--compile'] if c.get('compile') and not SMOKE else [])
    elif c.get('anchor') == 'stack':                      # p5_stack.py train (control without cancellation)
        deps.append('reg_w14_ss_v800')
        args = ['p5_stack.py', 'train', '--out', out]
        if c.get('prev'):
            p, pp = dep(c['prev']); deps.append(p); args += ['--prev', pp]
        else:
            args += ['--prev', '']
        for k in ('bs', 'lr', 'seed'):
            if k in c:
                args += [f'--{k}', str(c[k])]
        args += ['--steps', str(steps)] + (['--compile'] if c.get('compile') and not SMOKE else [])
    else:                                                 # otfs5.train
        args = ['-m', 'otfs5.train', '--out', out, '--mode', c['train_mode']]
        if c.get('teacher'):
            t, tp = dep(c['teacher']); deps.append(t); args += ['--teacher', tp, '--teacher_steps', c['teacher_steps']]
        for k in ('kd_alpha', 'layout', 'bs', 'lr', 'warmup', 'ema', 'model_channels', 'channel_mult', 'attention',
                  'num_res_blocks', 'speed_lo', 'speed_hi', 'seed', 'pred', 'schedule', 'anchor'):
            if k in c and c[k] is not None:
                args += [f'--{k}', str(c[k])]
        args += ['--paths', os.path.join(ROOT, 'paths'), '--steps', str(steps),
                 '--val_every', str(10 if SMOKE else c['val_every'])]
        for k in ('centre', 'single_stream'):
            if c.get(k):
                args.append(f'--{k}')
        if c.get('compile') and not SMOKE:
            args.append('--compile')
    return deps, out, args


def train(name, stack=()):
    if name in stack:
        raise SystemExit(f'dependency cycle at {name}')
    deps, out, args = argv_of(name, cfg_of(name))
    for d in deps:
        train(d, stack + (name,))
    if os.path.exists(os.path.join(out, 'DONE')):
        return
    os.makedirs(out, exist_ok=True)
    print(f'[train] {name}', flush=True)
    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS='1')
    with open(os.path.join(out, 'train.log'), 'a') as log:
        r = subprocess.run([PY] + args, cwd=os.path.join(ROOT, 'code'), stdout=log, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise SystemExit(f'training {name} failed, see {out}/train.log')
    open(os.path.join(out, 'DONE'), 'w').close()


if __name__ == '__main__':
    for n in sys.argv[1:]:
        train(n)

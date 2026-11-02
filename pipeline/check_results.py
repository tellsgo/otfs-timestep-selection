"""Compare the reproduced results with the reference results behind the paper.

  python pipeline/check_results.py            # every file in reference_results/ that also exists in results/v21/

For every pair of numbers at the same place in the two JSON files it reports the largest difference.  Values in dB
(keys with 'db' or 'nmse') are compared in dB, the others relatively.  Retrained networks and a different GPU give
small differences, typically a few hundredths to a tenth of a dB, the conclusions of the paper should not change.
"""
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF, NEW = os.path.join(ROOT, 'reference_results'), os.path.join(ROOT, 'results', 'v21')


def leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f'{path}[{i}]')
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        yield path, float(x)


def main():
    rows = []
    for f in sorted(os.listdir(REF)):
        if not f.endswith('.json'):
            continue
        if not os.path.exists(os.path.join(NEW, f)):
            rows.append((f, 'not reproduced yet', '', '')); continue
        a = dict(leaves(json.load(open(os.path.join(REF, f)))))
        b = dict(leaves(json.load(open(os.path.join(NEW, f)))))
        common = [k for k in a if k in b]
        db = [abs(a[k] - b[k]) for k in common if 'db' in k.lower() or 'nmse' in k.lower()]
        rel = [abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in common if not ('db' in k.lower() or 'nmse' in k.lower())]
        rows.append((f, f'{len(common)}/{len(a)} values', f'max |dB diff| {max(db):.3f}' if db else '',
                     f'max rel diff {max(rel):.3g}' if rel else ''))
    w = max(len(r[0]) for r in rows) if rows else 10
    for r in rows:
        print(f'{r[0]:<{w}}  {r[1]:<20} {r[2]:<22} {r[3]}')


if __name__ == '__main__':
    sys.exit(main())

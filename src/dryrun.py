"""
dryrun — 실험 파일을 데이터 없이 검증한다.

로컬 PC 에 1.1GB 데이터를 받지 않고도, 코랩에 올리기 전에
"이 실험이 문법·인자·조합 수 면에서 말이 되는가" 를 확인할 수 있다.

    python -m dryrun experiments/local/내실험.py
    LGA_DRYRUN=1 python experiments/local/내실험.py     (같은 효과)

검증하는 것
  · 문법 오류
  · run_experiment 인자 이름과 model/kind 값
  · 격자 조합 수와 예상 소요 시간
  · 원장에 쓰일 이름이 이미 있는지

검증하지 않는 것 (데이터가 있어야 알 수 있다)
  · 피처 함수가 실제로 도는지
  · 점수가 얼마나 나오는지
"""
import itertools
import json
import os
import sys

__all__ = ['dry_run', 'main']

_VALID_KIND = {'hparam', 'feature', 'blend'}
_VALID_MODEL = {'xgb', 'lgb', 'cb'}
# 조합 1개당 대략 소요(분). 시드 1개 기준, 폴드 2개 합산.
_MIN_PER_COMBO = {'xgb': 1.4, 'lgb': 1.0, 'cb': 2.2}

_calls = []


def _record(**kw):
    _calls.append(kw)


def dry_run(path, verbose=True):
    """실험 파일을 실행하되 run_experiment 를 가로채 검사만 한다."""
    import experiment as E

    real = E.run_experiment
    _calls.clear()

    def fake(name, kind='hparam', grid=None, base_params=None, features=None,
             seeds=2, runner='unknown', feature_fn=None, notes='',
             baseline=None, verbose=True, model='xgb', ref_params=None):
        _record(name=name, kind=kind, grid=grid, seeds=seeds,
                runner=runner, model=model, has_fn=feature_fn is not None)
        return None

    E.run_experiment = fake
    os.environ.setdefault('LGA_RUNNER', 'dryrun')
    try:
        src = open(path, encoding='utf-8').read()
        code = compile(src, path, 'exec')
        exec(code, {'__name__': '__main__', '__file__': path})
    finally:
        E.run_experiment = real

    problems, total_min = [], 0.0
    for c in _calls:
        if c['kind'] not in _VALID_KIND:
            problems.append(f"kind='{c['kind']}' 는 없는 값. {sorted(_VALID_KIND)} 중 하나여야 한다")
        if c['model'] not in _VALID_MODEL:
            problems.append(f"model='{c['model']}' 는 없는 값. {sorted(_VALID_MODEL)} 중 하나여야 한다")
        if c['kind'] == 'feature' and not c['has_fn']:
            problems.append(f"'{c['name']}': kind='feature' 인데 feature_fn 이 없다")
        g = c['grid'] or {}
        n = 1
        for v in g.values():
            n *= max(len(v), 1)
        mins = n * c['seeds'] * _MIN_PER_COMBO.get(c['model'], 1.5)
        total_min += mins
        if verbose:
            print(f"  {c['name']:28s} {c['model']:3s} {c['kind']:7s} "
                  f"조합 {n:3d} × 시드 {c['seeds']}  ≈ {mins:5.0f}분")
        if n * c['seeds'] > 60:
            problems.append(f"'{c['name']}': 조합×시드 {n*c['seeds']}개는 코랩 세션에 너무 많다 "
                            f"(GPU 제한에 걸린다). 나눠서 돌려라")

    if verbose:
        print(f"\nrun_experiment 호출 {len(_calls)}건, 총 ≈ {total_min:.0f}분")
        if problems:
            print('\n문제:')
            for p in problems:
                print(f'  ❌ {p}')
        else:
            print('✅ 문제 없음. push 하고 코랩 러너에서 돌려라.')
    return {'calls': _calls, 'problems': problems, 'minutes': total_min}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    r = dry_run(argv[0])
    return 1 if r['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())

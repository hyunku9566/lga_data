"""
config — 경로 설정. 로컬(/home/lee/lga)과 Colab(/content/lga) 양쪽에서 동작한다.

우선순위
  1. 환경변수 LGA_ROOT
  2. Colab 감지 시  /content/lga
  3. 그 외         /home/lee/lga

디렉토리 구조
  <ROOT>/data/       train.csv, test.csv, trackman_history.csv, sample_submission.csv
  <ROOT>/cache/      파생 캐시 (features/X98/aligned/tm5/oof_comp)  ※ git 제외
  <ROOT>/results/    실험 산출물                                    ※ git 제외
  <ROOT>/ledger/     팀원별 실험 원장 CSV (Drive 공유 폴더를 가리켜도 된다)
  <ROOT>/assets/     pitcher_map.csv, v6_tmsel.json 등 소용량 동봉 자산

주의: 레거시 로컬 경로에서는 캐시가 <ROOT> 바로 아래(X98.parquet 등)에 있다.
     resolve_cache() 가 두 위치를 모두 찾아보므로 기존 파일을 옮기지 않아도 된다.
"""
import os
import sys

__all__ = ['ROOT', 'DATA_DIR', 'CACHE_DIR', 'RESULTS_DIR', 'LEDGER_DIR', 'ASSETS_DIR',
           'IS_COLAB', 'DEV', 'path', 'resolve_cache', 'cache_path', 'ensure_dirs', 'describe']


def _detect_colab() -> bool:
    if os.environ.get('LGA_FORCE_COLAB') == '1':
        return True
    if 'google.colab' in sys.modules:
        return True
    return os.path.isdir('/content') and os.path.isdir('/usr/local/lib/python3.11/dist-packages')


IS_COLAB = _detect_colab()

_default_root = '/content/lga' if IS_COLAB else '/home/lee/lga'
ROOT = os.environ.get('LGA_ROOT', _default_root).rstrip('/') + '/'

DATA_DIR    = os.environ.get('LGA_DATA',    ROOT + 'data/')
CACHE_DIR   = os.environ.get('LGA_CACHE',   ROOT + 'cache/')
RESULTS_DIR = os.environ.get('LGA_RESULTS', ROOT + 'results/')
LEDGER_DIR  = os.environ.get('LGA_LEDGER',  ROOT + 'ledger/')
ASSETS_DIR  = os.environ.get('LGA_ASSETS',  ROOT + 'assets/')

# XGBoost 장치. 두 스크립트를 GPU 에 나눠 돌리려면 LGA_DEV=cuda:1 로 실행한다.
DEV = os.environ.get('LGA_DEV', 'cuda:0')

# 파생 캐시 파일명 (논리이름 -> 파일명)
CACHE_FILES = {
    'features': 'features.parquet',    # stage1 산출 (원본 피처 + 역산 + 트랙맨요약)
    'x98':      'X98.parquet',         # features + S1(투수x상황) + S2(매치업)
    'aligned':  'aligned.parquet',     # 트랙맨 투구단위 정렬
    'tm5':      'tm5.parquet',         # 트랙맨 18지표 (투수 as-of)
    'oof_comp': 'oof_comp.parquet',    # 성분 4종 OOF 로짓 + 파생 2
}

# 레거시 로컬 경로 (기존 파일을 그대로 재사용하기 위한 탐색 후보)
_LEGACY = {
    'features': [ROOT + 'features.parquet'],
    'x98':      [ROOT + 'X98.parquet'],
    'aligned':  [ROOT + 'aligned.parquet'],
    'tm5':      [ROOT + 'results14/tm5.parquet'],
    'oof_comp': ['/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/'
                 'scratchpad/oof_comp.parquet'],
}


def path(*parts) -> str:
    """ROOT 기준 경로 결합."""
    return os.path.join(ROOT, *parts)


def cache_path(key: str) -> str:
    """캐시의 표준 저장 위치 (새로 만들 때 쓰는 경로)."""
    if key not in CACHE_FILES:
        raise KeyError(f'알 수 없는 캐시 키: {key} (가능: {list(CACHE_FILES)})')
    return os.path.join(CACHE_DIR, CACHE_FILES[key])


def resolve_cache(key: str):
    """이미 존재하는 캐시를 찾는다. 표준 위치 -> 레거시 위치 순.
    없으면 None 을 돌려준다."""
    p = cache_path(key)
    if os.path.exists(p):
        return p
    for q in _LEGACY.get(key, []):
        if os.path.exists(q):
            return q
    return None


def ensure_dirs():
    for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, LEDGER_DIR, ASSETS_DIR):
        os.makedirs(d, exist_ok=True)


def asset(name: str) -> str:
    """소용량 동봉 자산 경로. assets/ 에 없으면 ROOT 바로 아래를 본다."""
    p = os.path.join(ASSETS_DIR, name)
    return p if os.path.exists(p) else os.path.join(ROOT, name)


def describe() -> str:
    lines = [f'IS_COLAB   {IS_COLAB}', f'ROOT       {ROOT}',
             f'DATA_DIR   {DATA_DIR}', f'CACHE_DIR  {CACHE_DIR}',
             f'RESULTS    {RESULTS_DIR}', f'LEDGER     {LEDGER_DIR}',
             f'DEV        {DEV}', '', '캐시 상태:']
    for k in CACHE_FILES:
        r = resolve_cache(k)
        if r:
            mb = os.path.getsize(r) / 2 ** 20
            lines.append(f'  {k:9s} OK   {mb:7.1f}MB  {r}')
        else:
            lines.append(f'  {k:9s} 없음      -> {cache_path(k)}')
    lines.append('')
    lines.append('원본 데이터:')
    for f in ('train.csv', 'test.csv', 'trackman_history.csv', 'sample_submission.csv'):
        p = os.path.join(DATA_DIR, f)
        lines.append(f'  {f:24s} {"OK" if os.path.exists(p) else "없음"}  {p}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe())

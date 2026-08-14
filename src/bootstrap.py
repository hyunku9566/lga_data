"""
bootstrap — 어느 노트북에서든 첫 셀 하나로 실험 준비를 끝낸다.

코랩은 노트북마다 런타임(VM)이 따로 뜬다. 01 노트북이 /content 에 받아둔
파일은 03 노트북을 열면 없다. 그래서 모든 노트북이 이걸 먼저 부른다.

    from src.bootstrap import setup
    setup(runner='홍길동')

하는 일
  1. Drive 마운트하고 경로 환경변수를 잡는다
  2. 필요한 파일을 확보한다 — 우선순위: 로컬 → Drive → 데이터 서버
  3. Drive 에 없던 파일은 받은 뒤 Drive 에 백업한다 (다음 런타임에서 재사용)

Drive 에 캐시가 있으면 30초~2분, 서버에서 새로 받으면 3~6분 걸린다.
"""
import os
import shutil
import sys
import time

__all__ = ['setup']

_NEED = [
    ('train.csv',             'data'),
    ('test.csv',              'data'),
    ('sample_submission.csv', 'data'),
    ('X98.parquet',           'cache'),
    ('features.parquet',      'cache'),
    ('tm5.parquet',           'cache'),
    ('oof_comp.parquet',      'cache'),
]
# 트랙맨 원본과 aligned 는 파생 캐시를 이미 받으면 실험에 필요 없다.
_OPTIONAL = [
    ('trackman_history.csv',  'data'),
    ('aligned.parquet',       'cache'),
]


def _mount_drive():
    try:
        from google.colab import drive
    except ImportError:
        return None
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
    return '/content/drive/MyDrive'


def setup(runner=None, drive_root=None, repo='/content/lga-repo',
          local_root='/content/lga', need_optional=False, verbose=True):
    """실험 준비를 끝내고 config 모듈을 돌려준다."""
    t0 = time.time()
    mydrive = _mount_drive()
    is_colab = mydrive is not None
    if drive_root is None:
        drive_root = f'{mydrive}/lga' if is_colab else None

    if is_colab:
        os.environ.setdefault('LGA_ROOT', local_root)
        os.environ.setdefault('LGA_DATA', f'{local_root}/data/')
        os.environ.setdefault('LGA_CACHE', f'{local_root}/cache/')
        os.environ.setdefault('LGA_ASSETS', f'{repo}/assets/')
        os.environ.setdefault('LGA_LEDGER', f'{drive_root}/ledger/')
        os.environ.setdefault('LGA_DEV', 'cuda:0')
        for d in (f'{drive_root}/data', f'{drive_root}/cache',
                  f'{drive_root}/ledger', f'{local_root}/data', f'{local_root}/cache'):
            os.makedirs(d, exist_ok=True)
    # 두 규약을 모두 지원한다.
    #   from src.lib_lga import ...   (레포 루트가 경로에 있을 때)
    #   import lib_lga                (src/ 가 경로에 있을 때)
    for p in (os.path.join(repo, 'src'), repo):
        if p not in sys.path:
            sys.path.insert(0, p)

    from . import config as C

    want = list(_NEED) + (list(_OPTIONAL) if need_optional else [])
    dst_of = {'data': C.DATA_DIR, 'cache': C.CACHE_DIR}
    missing = []
    from_drive = 0

    for name, kind in want:
        local = os.path.join(dst_of[kind], name)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            continue
        if drive_root:
            dr = os.path.join(drive_root, kind, name)
            if os.path.exists(dr) and os.path.getsize(dr) > 0:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                shutil.copy(dr, local)
                from_drive += 1
                if verbose:
                    print(f'  Drive → 로컬  {name}  {os.path.getsize(local)/2**20:.0f}MB')
                continue
        missing.append((name, kind))

    if missing:
        if verbose:
            print(f'\nDrive 에 없어 서버에서 받는다: {", ".join(n for n, _ in missing)}')
        from .download import fetch_all
        fetch_all(only=[n for n, _ in missing])
        # 다음 런타임을 위해 Drive 에 백업
        if drive_root:
            for name, kind in missing:
                src = os.path.join(dst_of[kind], name)
                if os.path.exists(src) and os.path.getsize(src) > 0:
                    shutil.copy(src, os.path.join(drive_root, kind, name))
            if verbose:
                print('  Drive 에 백업했다. 다음 런타임부터는 다시 안 받는다.')

    if runner:
        os.environ['LGA_RUNNER'] = runner

    if verbose:
        have = sum(1 for n, k in want
                   if os.path.exists(os.path.join(dst_of[k], n)))
        print(f'\n준비 완료  파일 {have}/{len(want)}  '
              f'(Drive 재사용 {from_drive}개)  {time.time()-t0:.0f}초')
        print(f'  ROOT   {C.ROOT}')
        print(f'  LEDGER {C.LEDGER_DIR}')
        if runner:
            print(f'  RUNNER {runner}')
    return C

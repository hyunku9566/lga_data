"""
download — 팀 데이터 서버에서 원본 데이터와 파생 캐시를 받아온다.

서버는 Basic Auth + Cloudflare Tunnel 뒤에 있고 Range(이어받기)를 지원한다.
비밀번호는 저장소에 넣지 않는다. 아래 셋 중 하나로 준다.

  1. 환경변수      os.environ['LGA_PASSWORD'] = '...'
  2. 인자          fetch_all(password='...')
  3. 대화형 입력    아무것도 안 주면 getpass 로 물어본다 (노트북에서 권장)

사용 예 (Colab):
    from src.download import fetch_all, check_reachable
    check_reachable()          # 먼저 도달 가능한지 확인
    fetch_all()                # 비밀번호 물어보고 전부 받기

주의: 한국 ISP 에서 이 도메인이 차단되는 사례가 확인됐다(warning.or.kr 로 리다이렉트).
     Colab 은 국내망이 아니라 대개 정상이지만, 로컬 PC 에서는 실패할 수 있다.
     check_reachable() 이 그 상황을 구분해서 알려준다.
"""
import os
import hashlib
import subprocess
import urllib.request
import urllib.error
import base64

BASE_URL = os.environ.get('LGA_DATA_URL', 'https://data.hyunku.mmv.kr')
USER = os.environ.get('LGA_DATA_USER', 'team')

# (파일명, 저장 위치 종류) — 'data' 는 DATA_DIR, 'cache' 는 CACHE_DIR, 'assets' 는 ASSETS_DIR
FILES = [
    ('train.csv',             'data'),
    ('test.csv',              'data'),
    ('sample_submission.csv', 'data'),
    ('trackman_history.csv',  'data'),
    ('X98.parquet',           'cache'),
    ('features.parquet',      'cache'),
    ('aligned.parquet',       'cache'),
    ('tm5.parquet',           'cache'),
    ('oof_comp.parquet',      'cache'),
    ('pitcher_map.csv',       'assets'),
    ('v6_tmsel.json',         'assets'),
]

__all__ = ['check_reachable', 'fetch_all', 'fetch_one', 'verify', 'BASE_URL']


def _get_password(password=None):
    if password:
        return password
    env = os.environ.get('LGA_PASSWORD')
    if env:
        return env
    from getpass import getpass
    return getpass(f'{USER}@{BASE_URL} 비밀번호: ')


def _dest_dir(kind):
    from . import config
    return {'data': config.DATA_DIR, 'cache': config.CACHE_DIR,
            'assets': config.ASSETS_DIR}[kind]


def check_reachable(timeout=15):
    """서버 도달 가능 여부를 진단한다. 차단 상황을 구분해서 알려준다."""
    try:
        req = urllib.request.Request(BASE_URL + '/', method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(400).decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f'✅ 서버 정상 (401 = 인증 필요). {BASE_URL}')
            return True
        print(f'⚠️  HTTP {e.code} — 서버는 응답하나 예상과 다르다')
        return False
    except Exception as e:
        print(f'❌ 연결 실패: {type(e).__name__}: {e}')
        print('   국내 ISP 차단이거나 터널이 내려갔을 수 있다.')
        print('   Colab 이 아닌 로컬 PC 라면 Colab 에서 다시 시도해봐라.')
        return False

    if 'warning.or.kr' in body:
        print('❌ 국내 ISP 차단 페이지가 반환됐다 (warning.or.kr).')
        print('   서버 문제가 아니라 네트워크 경로 문제다. Colab 에서는 대개 정상이다.')
        return False
    print(f'⚠️  인증 없이 200 이 반환됐다. Basic Auth 가 걸려 있는지 확인해라.')
    return True


def fetch_one(name, kind, password, force=False, quiet=False):
    dest = os.path.join(_dest_dir(kind), name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force:
        if not quiet:
            print(f'  건너뜀 (이미 있음)  {name}')
        return dest
    url = f'{BASE_URL}/{name}'
    # wget -c 로 이어받기. 끊겨도 재실행하면 이어진다.
    cmd = ['wget', '-c', '-q', '--show-progress', '--progress=bar:force:noscroll',
           '--user', USER, '--password', password, '-O', dest, url]
    subprocess.run(cmd, check=True)
    return dest


def fetch_all(password=None, force=False, only=None):
    """전체 다운로드. only=['train.csv', ...] 로 일부만 받을 수 있다."""
    password = _get_password(password)
    targets = [(n, k) for n, k in FILES if only is None or n in only]
    print(f'{len(targets)}개 파일 다운로드 — {BASE_URL}')
    for name, kind in targets:
        print(f'▶ {name}')
        fetch_one(name, kind, password, force=force)
    # MANIFEST 는 항상 최신으로 받는다
    from . import config
    man = os.path.join(config.ROOT, 'MANIFEST.txt')
    try:
        fetch_one('MANIFEST.txt', 'data', password, force=True, quiet=True)
        man = os.path.join(config.DATA_DIR, 'MANIFEST.txt')
    except Exception as e:
        print(f'  MANIFEST 를 받지 못했다: {e}')
        return
    print('\n체크섬 검증')
    verify(man)


def verify(manifest_path):
    """MANIFEST.txt 의 sha256 과 실제 파일을 대조한다."""
    kinds = dict(FILES)
    ok = bad = missing = 0
    with open(manifest_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3 or parts[2] == 'SHA256' or parts[0].startswith('-'):
                continue
            name, _size, want = parts
            kind = kinds.get(name)
            if kind is None:
                continue
            p = os.path.join(_dest_dir(kind), name)
            if not os.path.exists(p):
                print(f'  없음  {name}'); missing += 1; continue
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            if h.hexdigest() == want:
                print(f'  OK    {name}'); ok += 1
            else:
                print(f'  불일치 {name}  — 다시 받아라 (force=True)'); bad += 1
    print(f'\n정상 {ok} / 불일치 {bad} / 없음 {missing}')
    return bad == 0 and missing == 0

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

BASE_URL = os.environ.get('LGA_DATA_URL', 'https://lgadata.hyunku.mmv.kr')
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

__all__ = ['check_reachable', 'probe', 'fetch_all', 'fetch_one', 'verify', 'BASE_URL']


def _get_password(password=None):
    if password:
        return password
    env = os.environ.get('LGA_PASSWORD')
    if env:
        return env
    from getpass import getpass
    return getpass(f'{USER}@{BASE_URL} 비밀번호: ')


def _dest_dir(kind):
    try:
        from . import config
    except ImportError:
        import config
    return {'data': config.DATA_DIR, 'cache': config.CACHE_DIR,
            'assets': config.ASSETS_DIR}[kind]


# Cloudflare 는 파이썬 기본 User-Agent 를 봇으로 보고 403 을 준다.
# 브라우저 UA 를 쓰면 통과한다. wget 도 동일하게 맞춘다.
UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


def check_reachable(timeout=20):
    """서버 도달 가능 여부를 진단한다. 차단 상황을 구분해서 알려준다."""
    try:
        req = urllib.request.Request(BASE_URL + '/', method='GET',
                                     headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(400).decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f'✅ 서버 정상 (401 = 인증 필요). {BASE_URL}')
            return True
        if e.code == 403:
            print('⚠️  HTTP 403 — Cloudflare 가 요청을 차단했다.')
            print('   보통 봇 탐지다. 아래를 확인해라:')
            print('   · Cloudflare 대시보드 > Security > Bot Fight Mode 를 끈다')
            print('   · 또는 WAF 규칙에서 이 호스트를 예외 처리한다')
            print(f'   터미널에서 이게 401 이면 서버는 정상이다:')
            print(f'     !curl -sS -o /dev/null -w "%{{http_code}}" {BASE_URL}/MANIFEST.txt')
            return False
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
    print('⚠️  인증 없이 200 이 반환됐다. Basic Auth 가 걸려 있는지 확인해라.')
    return True


def probe(name='MANIFEST.txt', password=None, timeout=25):
    """단일 파일의 HTTP 상태만 확인한다. 실패 원인 파악용."""
    password = _get_password(password)
    url = f'{BASE_URL}/{name}'
    tok = base64.b64encode(f'{USER}:{password}'.encode()).decode()
    # HEAD 는 쓰지 않는다. Cloudflare/nginx 가 HEAD 를 다르게 처리해
    # 같은 자격증명인데도 401 을 주는 경우가 있었다.
    # 대신 GET 에 Range 를 붙여 1바이트만 받는다.
    req = urllib.request.Request(url, method='GET',
                                 headers={'User-Agent': UA,
                                          'Authorization': f'Basic {tok}',
                                          'Range': 'bytes=0-0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(1)
            # 206 이면 Content-Range 에 전체 크기가 들어있다
            cr = r.headers.get('Content-Range')
            size = cr.split('/')[-1] if cr and '/' in cr else r.headers.get('Content-Length')
            if size and size.isdigit():
                print(f'✅ {name}  HTTP {r.status}  크기 {int(size)/2**20:.1f}MB')
            else:
                print(f'✅ {name}  HTTP {r.status}')
            return 200
    except urllib.error.HTTPError as e:
        hint = {401: '비밀번호가 틀렸다',
                403: 'Cloudflare 가 막았다 (Bot Fight Mode 확인)',
                404: '서버에 그 파일이 없다 — 아직 업로드 안 된 것 같다'}.get(e.code, '')
        print(f'❌ {name}  HTTP {e.code}  {hint}')
        return e.code
    except Exception as e:
        print(f'❌ {name}  {type(e).__name__}: {e}')
        return None


def _netrc(password):
    """비밀번호를 명령줄에 노출하지 않기 위해 임시 netrc 를 쓴다.
       (argv 는 traceback 과 ps 에 그대로 찍힌다)"""
    import tempfile
    host = BASE_URL.split('://', 1)[-1].split('/')[0]
    fd, path = tempfile.mkstemp(prefix='.netrc_')
    with os.fdopen(fd, 'w') as f:
        f.write(f'machine {host} login {USER} password {password}\n')
    os.chmod(path, 0o600)
    return path


def _expected_size(name):
    """MANIFEST.txt 가 이미 있으면 기대 크기를 돌려준다."""
    try:
        from . import config
    except ImportError:
        import config
    for man in (os.path.join(config.DATA_DIR, 'MANIFEST.txt'),
                os.path.join(config.ROOT, 'MANIFEST.txt')):
        if not os.path.exists(man):
            continue
        with open(man) as f:
            for line in f:
                p = line.split()
                if len(p) == 3 and p[0] == name and p[1].isdigit():
                    return int(p[1])
    return None


def fetch_one(name, kind, password, force=False, quiet=False, netrc=None):
    dest = os.path.join(_dest_dir(kind), name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force:
        have = os.path.getsize(dest)
        want = _expected_size(name)
        if have == 0:
            print(f'  빈 파일이라 다시 받는다  {name}')
            os.remove(dest)
        elif want is not None and have != want:
            print(f'  크기 불일치({have:,} != {want:,}) 이어받는다  {name}')
        else:
            if not quiet:
                print(f'  건너뜀 (이미 있음)  {name}')
            return dest
    url = f'{BASE_URL}/{name}'
    own = netrc is None
    if own:
        netrc = _netrc(password)
    try:
        # curl 을 쓴다. GNU wget 에는 --netrc-file 옵션이 없다(그건 curl 것이다).
        #   -C -            이어받기. 끊겨도 재실행하면 이어진다
        #   --netrc-file    자격증명을 argv 에 남기지 않는다
        #   -f              HTTP 에러면 실패로 처리
        cmd = ['curl', '-fL', '-C', '-', '--retry', '3', '--retry-delay', '2',
               '--netrc-file', netrc, '-A', UA,
               '--progress-bar', '-o', dest, url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 33:
            # 서버가 Range 를 거부하면 처음부터 다시 받는다
            if os.path.exists(dest):
                os.remove(dest)
            cmd.remove('-C'); cmd.remove('-')
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or '').strip().splitlines()[-4:]
            print(f'  ❌ {name} 실패 (curl {r.returncode})')
            for l in tail:
                print(f'     {l}')
            if os.path.exists(dest) and os.path.getsize(dest) == 0:
                os.remove(dest)
            raise RuntimeError(f'{name} 다운로드 실패')
        sz = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f'     받음 {sz/2**20:.1f}MB')
    finally:
        if own and os.path.exists(netrc):
            os.remove(netrc)
    return dest


def fetch_all(password=None, force=False, only=None, stop_on_error=False):
    """전체 다운로드. only=['train.csv', ...] 로 일부만 받을 수 있다.

    한 파일이 실패해도 나머지를 계속 받는다 (stop_on_error=True 면 중단).
    마지막에 실패 목록과 체크섬 검증 결과를 보여준다."""
    password = _get_password(password)
    targets = [(n, k) for n, k in FILES if only is None or n in only]
    print(f'{len(targets)}개 파일 다운로드 — {BASE_URL}\n')
    nrc = _netrc(password)
    failed = []
    try:
        for name, kind in targets:
            print(f'▶ {name}')
            try:
                fetch_one(name, kind, password, force=force, netrc=nrc)
            except Exception as e:
                failed.append(name)
                if stop_on_error:
                    raise
        try:
            fetch_one('MANIFEST.txt', 'data', password, force=True,
                      quiet=True, netrc=nrc)
        except Exception:
            failed.append('MANIFEST.txt')
    finally:
        if os.path.exists(nrc):
            os.remove(nrc)

    if failed:
        print(f'\n실패 {len(failed)}건: {", ".join(failed)}')
        print('원인을 보려면:  from src.download import probe; probe("train.csv")')
    try:
        from . import config
    except ImportError:
        import config
    man = os.path.join(config.DATA_DIR, 'MANIFEST.txt')
    if os.path.exists(man):
        print('\n체크섬 검증')
        verify(man)
    return not failed


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

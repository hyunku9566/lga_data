"""
sync — 실험 원장을 git 으로 팀과 공유한다.

각자 자기 Drive 를 쓰면 원장이 각자 Drive 에만 쌓여 집계가 안 된다.
Drive 폴더를 서로 공유하는 대신 **git 으로 모은다.**
파일명이 사람마다 다르므로(ledger_<이름>.csv) 충돌이 나지 않는다.

코랩에서 푸시하려면 GitHub 토큰이 필요하다. 한 번만 설정하면 된다.

  코랩 왼쪽 🔑(보안 비밀) → 새 보안 비밀 추가
      이름  GITHUB_TOKEN
      값    github.com > Settings > Developer settings >
            Personal access tokens > Fine-grained tokens 에서 발급
            (이 저장소에 Contents: Read and write 권한만 주면 된다)
      노트북 액세스  켜기
"""
import os
import shutil
import subprocess

__all__ = ['push_ledger', 'pull_ledgers', 'repo_ledger_dir']

REPO = os.environ.get('LGA_REPO', '/content/lga-repo')
REMOTE = 'https://github.com/hyunku9566/lga_data.git'


def repo_ledger_dir():
    d = os.path.join(REPO, 'ledgers')
    os.makedirs(d, exist_ok=True)
    return d


def _token():
    tok = os.environ.get('GITHUB_TOKEN')
    if tok:
        return tok
    try:
        from google.colab import userdata
        return userdata.get('GITHUB_TOKEN')
    except Exception:
        return None


def _git(*args, check=False):
    r = subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} 실패:\n{r.stderr.strip()}')
    return r


def pull_ledgers(quiet=False):
    """다른 팀원이 올린 원장을 받아온다."""
    r = _git('pull', '--rebase', '-q', 'origin', 'main')
    if r.returncode != 0 and not quiet:
        print(f'  원장 최신화 실패(무시하고 진행): {r.stderr.strip().splitlines()[-1:]}')
    return repo_ledger_dir()


def push_ledger(runner, ledger_dir=None, branch='main'):
    """내 원장을 저장소에 커밋·푸시한다. 토큰이 없으면 건너뛴다."""
    try:
        import config as C
    except ImportError:
        from . import config as C
    ledger_dir = ledger_dir or C.LEDGER_DIR
    safe = ''.join(ch for ch in str(runner) if ch.isalnum() or ch in '-_') or 'unknown'
    src = os.path.join(ledger_dir, f'ledger_{safe}.csv')
    if not os.path.exists(src):
        print('  올릴 원장이 없다 (실험을 먼저 돌려라)')
        return False

    tok = _token()
    if not tok:
        print('  GITHUB_TOKEN 이 없어 원장을 올리지 못했다.')
        print('  코랩 왼쪽 🔑 에서 GITHUB_TOKEN 을 등록하면 자동으로 공유된다.')
        print(f'  당장은 이 파일을 직접 올려도 된다: {src}')
        return False

    dst = os.path.join(repo_ledger_dir(), f'ledger_{safe}.csv')
    shutil.copy(src, dst)

    _git('config', 'user.name', safe)
    _git('config', 'user.email', f'{safe}@lga.local')
    _git('add', 'ledgers/')
    r = _git('diff', '--cached', '--quiet')
    if r.returncode == 0:
        print('  원장에 새 내용이 없다')
        return True

    n = sum(1 for _ in open(dst, encoding='utf-8-sig')) - 1
    _git('commit', '-q', '-m', f'원장 갱신: {runner} ({n}건)')
    auth = REMOTE.replace('https://', f'https://{tok}@')
    for attempt in (1, 2):
        r = subprocess.run(['git', '-C', REPO, 'push', '-q', auth, f'HEAD:{branch}'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f'  ✅ 원장 {n}건을 저장소에 올렸다')
            return True
        if attempt == 1:
            _git('pull', '--rebase', '-q', auth, branch)
    print(f'  ❌ 푸시 실패: {r.stderr.strip().splitlines()[-1:]}')
    return False

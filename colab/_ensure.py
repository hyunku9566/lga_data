"""
_ensure — 프리셋 셀이 실행 순서에 의존하지 않게 해주는 보조 스크립트.

각 실험 셀 맨 위에 아래 한 줄이 들어있다.

    if 'E' not in globals(): exec(open('/content/lga-repo/colab/_ensure.py').read())

부트스트랩 셀을 안 돌리고 중간 셀부터 실행해도 필요한 것만 채워준다.
런타임이 끊겨 변수가 날아간 뒤 이어서 돌릴 때도 같다.
"""
import os
import sys

for _p in ('/content/lga-repo/src', '/content/lga-repo'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists('/content/lga-repo/src/config.py'):
    raise RuntimeError(
        '레포가 없다. 맨 위 부트스트랩 셀을 먼저 실행해라.\n'
        '(런타임 > 모두 실행 을 쓰면 순서가 보장된다)')

import config as C          # noqa: E402
import lib_lga as L         # noqa: E402
import experiment as E      # noqa: E402

RUNNER_NAME = os.environ.get('LGA_RUNNER', '').strip()
if not RUNNER_NAME:
    raise RuntimeError(
        '실행자 이름이 없다. 맨 위 부트스트랩 셀에서 RUNNER_NAME 을 채우고 실행해라.\n'
        "급하면 이 셀 위에서:  import os; os.environ['LGA_RUNNER']='본인이름'")

SEEDS = int(os.environ.get('LGA_SEEDS', '3'))

print(f'준비됨 · 실행자 {RUNNER_NAME} · 시드 {SEEDS} · ROOT {C.ROOT}')

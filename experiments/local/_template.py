"""
실험 파일 템플릿 — 이 파일을 복사해서 쓴다.

파일명 규칙:  <본인이름>_<주제>.py     예) yun_lgb_leaves.py

작업 순서
  1. 로컬에서 이 파일을 복사·수정
  2. 로컬 검증 (데이터 없어도 된다)
         python src/dryrun.py experiments/local/yun_lgb_leaves.py
  3. git push
  4. 코랩에서 colab/00_runner.ipynb 을 열고 SCRIPT 에 파일 경로만 적고 실행
  5. 결과는 원장에 쌓인다. 04_summary.ipynb 로 집계
"""
import os
import experiment as E

RUNNER = os.environ['LGA_RUNNER']      # 러너 노트북이 넣어준다. 건드리지 마라.

# ── 여기부터 수정 ────────────────────────────────────────────

E.run_experiment(
    name='내_실험_이름',               # 원장에 남는 키. 겹치지 않게
    kind='hparam',                     # 'hparam' | 'feature' | 'blend'
    model='xgb',                       # 'xgb' | 'lgb' | 'cb'
    grid={'max_depth': [6, 8], 'min_child_weight': [1500, 6000]},
    seeds=3,
    runner=RUNNER,
    notes='무엇을 왜 보려는지 한 줄',
)

# ── 참고: 이미 측정된 것 (다시 하지 마라) ──────────────────────
#   XGB  더 크게·더 느리게가 좋다. d10/mcw6000/n2000/lr0.005 부근
#   LGB  XGB 와 반대로 작을수록 좋다. leaves31 796.3 → leaves15 830.8
#   CB   깊을수록 붕괴한다. d6 745 → d8 640 → d10 437
#
# 기각된 것: 상황별 세분화 피처, 타깃 분해, 구종 예측 스태킹,
#           콜드스타트 표시, 2024 가중 강화, dart, exact, linear_tree
#
# 신호는 투수 정체성에 몰려 있다 (투수ID 797 vs 볼카운트 31, 주자상황 3.9).
# 상황 변수 쪽 아이디어는 대부분 실패했다.

# 로컬에서 작성하는 실험 파일

노트북 셀을 고치지 않는다. **여기에 파이썬 파일을 하나 만들고, 코랩 러너는 그 파일을 실행만 한다.**

## 왜 이렇게 하나

노트북 안에서 코드를 고치면 이런 것들이 계속 문제가 된다.

- 코랩은 GitHub 노트북을 **스냅샷으로** 연다. 저장소를 고쳐도 열려 있는 화면은 안 바뀐다
- 셀 실행 순서에 따라 결과가 달라진다
- 여러 명이 같은 노트북을 고치면 충돌한다

파일로 빼면 **git 이 관리하고, 리뷰가 되고, 재현이 된다.**

## 흐름

```
로컬에서 파일 작성
  → python src/dryrun.py experiments/local/내파일.py     (데이터 없이 검증)
  → git push
  → 코랩 00_runner.ipynb 에서 파일 경로만 지정하고 실행
  → 원장에 결과 → 04_summary.ipynb 로 집계
```

**로컬에 데이터를 받을 필요가 없다.** `dryrun` 은 데이터 없이 인자·격자·소요시간만 검사한다.

## 검증이 잡아주는 것

```
$ python src/dryrun.py experiments/local/yun_lgb.py
  lgb_leaves_sweep            lgb hparam  조합   6 × 시드 3  ≈    18분

run_experiment 호출 1건, 총 ≈ 18분
✅ 문제 없음. push 하고 코랩 러너에서 돌려라.
```

- 문법 오류
- `kind` / `model` 오타
- `kind='feature'` 인데 `feature_fn` 을 안 넘긴 경우
- **조합×시드가 너무 많아 코랩 GPU 제한에 걸릴 경우**

## 파일명 규칙

`<본인이름>_<주제>.py` — 예: `yun_lgb_leaves.py`, `jeon_cmp_retune.py`

이름이 겹치면 원장에서 누가 뭘 돌렸는지 구분이 안 된다.

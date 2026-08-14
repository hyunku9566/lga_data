# 제출물 빌드 · 추론 코드

리더보드에 실제로 올린 코드다. Phase 3 학습 코드 제출에도 이걸 낸다.

> **경로 주의.** 이 스크립트들은 개발 PC 경로(`/home/lee/lga/`)가 하드코딩돼 있다.
> 실험 결과를 그대로 재현하기 위해 **원본을 손대지 않고** 옮겨왔다.
> 코랩에서 돌리려면 `src/config.py` 를 쓰는 `src/prepare_data.py` 쪽을 봐라.

## 빌드 체인

```
build_v7.py    성분 라벨 역산 → 성분모델 4개 학습·저장 → 성분 OOF 생성
               → XGB/LGB 학습 → submit_v7/
build_v9.py    v8 에 CatBoost 축 추가 → submit_v9/
build_v10.py   재튜닝된 하이퍼파라미터로 XGB/LGB 재학습 → submit_v10/
script_v9.py   평가 서버가 실행하는 추론 진입점 (submit_v*/script.py 로 복사됨)
```

`build_v10.py` 는 `submit_v9/` 를 복사한 뒤 XGB/LGB 만 다시 학습한다.
CatBoost·성분모델·NN 가중치는 그대로 재사용한다.

## 현재 최고 제출물 = v10a

**v10a 는 별도 빌드 스크립트가 없다.** `build_v10.py` 산출물에서
블렌드 가중치만 v9 값으로 되돌린 것이다.

```
XGB .45 / LGB .35 / CB .20 / NN .30,  mt_alpha 0.4,  drift -0.020
```

`submit_v10/model/consts.json` 이 이미 이 값으로 맞춰져 있다.
자세한 내역은 저장소 루트의 `CURRENT_BEST.md` 참고.

## 추론 파이프라인 (`script_v9.py`)

```
test.csv
  → featurize()            원본 48컬럼 → 114 base 피처
  → 성분모델 4개 예측         cmp_reverse/middle/ball/strike → 로짓 6개
  → 120 피처 완성
  → XGB 12 / LGB 7 / CatBoost 1 / NN 16   각 축 예측
  → 4축 로짓 가중 평균
  → 드리프트 로짓 시프트 (-0.020)
  → submission.csv
```

600초 예산 안에서 돈다. 22.3만 행 기준 로컬 101초, 평가 서버 2분 14초 실측.

## 규칙 준수

- 성분·구종 라벨 역산은 **학습 시 보조 타깃으로만** 쓴다. 추론 입력에 넣지 않는다
- 평가 데이터 행 간 참조 없음. 감사 절차와 통과 증거는 `docs/RULES.md` 참고
- 드리프트 상수는 학습 데이터의 시즌 추세로만 정한다

# 실험 색인

로컬에서 돌린 실험 스크립트와 **결과 로그 전부**를 여기에 모았다.

- `run*.py` — 현행 프로토콜(`src/lib_lga.py`) 기준 실험 59건
- `archive/` — 그 이전 구버전 실험 18건 (경로·규약이 달라 그대로는 안 돌아간다)
- `logs/` — 결과 로그 287개. 파일명은 `<결과디렉토리>__<원본파일명>` 이다

**읽는 법**: 스크립트 상단 docstring 에 "무엇을 왜 쟀는가" 와 판정이 적혀 있다.
수치는 `logs/` 에서 확인한다. 종합 결론은 [`../docs/CEILING.md`](../docs/CEILING.md) 와
[`../docs/FINDINGS.md`](../docs/FINDINGS.md) 에 있다.

---

## 현행 실험

| 스크립트 | 내용 | 로그 |
|---|---|---|
| [`run20.py`](run20.py) | 20차 — 구종 라벨 역산의 값어치 측정 (근본 재검토) | `logs/results20__*` (2개) |
| [`run21.py`](run21.py) | 21차 — 타깃 인수분해 (지금까지 안 해본 마지막 구조적 아이디어) | `logs/results21__*` (2개) |
| [`run22.py`](run22.py) | 22차 — 병목 진단: 모델인가 데이터인가 | `logs/results22__*` (2개) |
| [`run23.py`](run23.py) | 23차 — 하이퍼파라미터 재탐색 (22차에서 과소적합 확인됨) | `logs/results23__*` (3개) |
| [`run24.py`](run24.py) | 24차 — 시간 일반화 3종 (22차 진단이 가리킨 방향) | `logs/results24__*` (3개) |
| [`run25.py`](run25.py) | 25차 — ABS regime 대응 (가장 중요한 구조 문제) | `logs/results25__*` (2개) |
| [`run26.py`](run26.py) | 26차 — LGB / CatBoost 재탐색 (마지막 미탐색 영역) | `logs/results26__*` (3개) |
| [`run27.py`](run27.py) | 27차 — 자원 제약 없는 '무겁고 구조적으로 다른' 학습기 탐색 | `logs/results27__*` (3개) |
| [`run28.py`](run28.py) | 28차 — v10 구성 확정용 최종 확인 | `logs/results28__*` (3개) |
| [`run29_cmp_retune.py`](run29_cmp_retune.py) | 29차 — 성분모델(cmp_*) 재튜닝 | `logs/results29__*` (3개) |
| [`run30_configdiv.py`](run30_configdiv.py) | 30차 — 시드 다양성 vs 설정 다양성 | `logs/results30__*` (2개) |
| [`run31_blend2fold.py`](run31_blend2fold.py) | 31차-B — 블렌드 가중치 질문을 '두 폴드 동시'로 재판정 (CPU 전용, 재학습 없음) | `logs/results31__*` (7개) |
| [`run31_blend_newhpo.py`](run31_blend_newhpo.py) | 31차-D — 출시본 HPO 로 블렌드 재판정 + HPO x 블렌드 상호작용 검증 | `logs/results31__*` (7개) |
| [`run31_fold2023.py`](run31_fold2023.py) | 31차 — 폴드2023 보조 검증 (선택 편향 진단) | `logs/results31__*` (7개) |
| [`run31_newhpo_folds.py`](run31_newhpo_folds.py) | 31차-C — 실제 출시된 모델(v10 HPO)의 폴드2022/2023 예측 생성 | `logs/results31__*` (7개) |
| [`run32_lgb_configdiv.py`](run32_lgb_configdiv.py) | 32차 — LightGBM 설정 다양성(Configuration Diversity) 검증 | `logs/results32__*` (1개) |
| [`run33_count_interactions.py`](run33_count_interactions.py) | 33차 — 볼카운트 및 위기상황 상호작용 피처(Count & Stress Interaction Features) 검증 | `logs/results33__*` (2개) |
| [`run34_pitch_conditioning.py`](run34_pitch_conditioning.py) | 34차 — 구종별(Fastball / Breaking / Offspeed) 제구 분해 및 엔트로피 피처 검증 | `logs/results34__*` (2개) |
| [`run35_catboost_tuning.py`](run35_catboost_tuning.py) | 35차 — CatBoost 아키텍처 다양성 및 시드 앙상블 검증 | `logs/results35__*` (1개) |
| [`run36_brier_objective.py`](run36_brier_objective.py) | 36차 — 트리를 Brier(제곱오차)로 직접 최적화하면 나은가 | `logs/results36__*` (1개) |
| [`run37_residual_decomp.py`](run37_residual_decomp.py) | 37차 — 현재 모델 잔차에 구조가 남아 있는가 (천장 판별) | — |
| [`run38_pitchtype_v2.py`](run38_pitchtype_v2.py) | 38차 — 구종 예측 모델 재구축 (옛 하이퍼파라미터로 남아 있던 마지막 축) | `logs/results38__*` (5개) |
| [`run39_shr_sweep.py`](run39_shr_sweep.py) | 39차 — 투수x상황 축소강도(SHR) 스윕 + 표본수 피처 | `logs/results39__*` (9개) |
| [`run40_trackman_residual_audit.py`](run40_trackman_residual_audit.py) | 40차 — TrackMan 고정 프로필의 시간외 잔차 신호 감사 (CPU 전용). | `logs/results40__*` (2개) |
| [`run41_exact_count.py`](run41_exact_count.py) | 41차 — 투수 × 정확 볼카운트(12개) 계층적 과거 이력 검증 | `logs/results41__*` (2개) |
| [`run42_exact_count_comp.py`](run42_exact_count_comp.py) | 42차 — 투수 × 정확 볼카운트(12개) 4대 물리 성분 이력 확장 검증 | `logs/results42__*` (2개) |
| [`run43_blend_diag.py`](run43_blend_diag.py) | 43차 — 왜 CV 이득이 LB 로 전이되지 않는가 | `logs/results43__*` (2개) |
| [`run44_tm_countmix.py`](run44_tm_countmix.py) | 44차 — 투수 x 볼카운트별 구종배합 / 물리 프로파일  (GPU0) | `logs/results44__*` (1개) |
| [`run45_batter_axis.py`](run45_batter_axis.py) | 45차 — 타자축 및 교차 이력  (GPU1) | `logs/results45__*` (2개) |
| [`run46_b3_confirm.py`](run46_b3_confirm.py) | 46차 — B3(투수 x 타자손 x 카운트) 확정 검증 | `logs/results46__*` (1개) |
| [`run47_pbc_comp.py`](run47_pbc_comp.py) | 47차 — pbc 키를 성분 라벨에 적용  (GPU0) | `logs/results47__*` (2개) |
| [`run48_pbc_variants.py`](run48_pbc_variants.py) | 48차 — pbc 계열 변형  (GPU1) | `logs/results48__*` (2개) |
| [`run49_lgb_retune.py`](run49_lgb_retune.py) | 49차 — LightGBM 재탐색 (블렌드 주력 축이 된 뒤 처음) | `logs/results49__*` (22개) |
| [`run50_lgb_stage2.py`](run50_lgb_stage2.py) | 50차 (49차 2단계) — LightGBM 재탐색 (블렌드 주력 축이 된 뒤 처음) | `logs/results50__*` (23개) |
| [`run51_lgb_stage3.py`](run51_lgb_stage3.py) | 51차 (49차 3단계 — extra_trees 기준) — LightGBM 재탐색 (블렌드 주력 축이 된 뒤 처음) | `logs/results51__*` (19개) |
| [`run52_cb_retune.py`](run52_cb_retune.py) | 52차 — CatBoost 재탐색 (블렌드 3번째 축, 한 번도 제대로 튜닝 안 함) | `logs/results52__*` (17개) |
| [`run53_et_axis.py`](run53_et_axis.py) | 53차 — ExtraTrees 를 5번째 블렌드 축으로 | `logs/results53__*` (2개) |
| [`run54_cb_stage2.py`](run54_cb_stage2.py) | 54차 — CatBoost 2단계 (depth4 + bagging_temperature 조합) (블렌드 3번째 축, 한 번도 제대로 튜닝 안 함) | `logs/results54__*` (15개) |
| [`run55_lgb_stage4.py`](run55_lgb_stage4.py) | 55차 (LGB 4단계 — leaves15+ET 기준) — LightGBM 재탐색 (블렌드 주력 축이 된 뒤 처음) | `logs/results55__*` (17개) |
| [`run56_blend_verify.py`](run56_blend_verify.py) | 56차 — 새 LGB/CB 설정의 블렌드 수준 검증 | `logs/results56__*` (1개) |
| [`run57_fold2022_contam.py`](run57_fold2022_contam.py) | 57차 — 폴드2022 는 regime 때문이 아니라 F리그 오염 때문에 이상했던 것인가 | `logs/results57__*` (2개) |
| [`run58_platoon.py`](run58_platoon.py) | 58차 — 투수 x 타자손(플래툰) 전용 계층 피처 | `logs/results58__*` (2개) |
| [`run59_pbc_rehier.py`](run59_pbc_rehier.py) | 59차 — pbc_* 의 축소 경로를 고친다 (병렬 추가가 아니라 대체) | `logs/results59__*` (2개) |
| [`run60_hindsight_probe.py`](run60_hindsight_probe.py) | 60차 — [진단 전용, 제출 금지] test 내부 투수별 집계의 값어치를 정량화 | `logs/results60__*` (2개) |
| [`run61_regimefold.py`](run61_regimefold.py) | 61차 — regime 일치 폴드: 2024 를 학습에 넣고 2024 를 맞힌다 | `logs/results61__*` (1개) |
| [`run62_statespace.py`](run62_statespace.py) | 62차 — 상태공간 투수능력 추정기 (기존 GBDT 구조와 무관한 별도 모형) | `logs/results62__*` (1개) |
| [`run63_hier.py`](run63_hier.py) | 63차 — 처음부터 다시: 계층 베이즈 모형 (기존 124피처 파이프라인 미사용) | `logs/results63__*` (3개) |
| [`run63b_hier.py`](run63b_hier.py) | 63차b — 계층 베이즈, 각 관측을 '그 시절 리그 수준'으로 디민 후 수축 | `logs/results63__*` (3개) |
| [`run63c_hier.py`](run63c_hier.py) | 63차b — 계층 베이즈, 각 관측을 '그 시절 리그 수준'으로 디민 후 수축 | `logs/results63__*` (3개) |
| [`run64_fclean.py`](run64_fclean.py) | 64차 — 1군/2군 오염 보정 (주최측 asof 는 R+F 를 섞어 누적한다) | `logs/results64__*` (2개) |
| [`run65_dropF.py`](run65_dropF.py) | 65차 — F(2군) 를 학습에서 전부 뺀다 | `logs/results65__*` (2개) |
| [`run66_fclean2.py`](run66_fclean2.py) | 64차 — 1군/2군 오염 보정 (주최측 asof 는 R+F 를 섞어 누적한다) | `logs/results66__*` (2개) |
| [`run67_prior.py`](run67_prior.py) | 67차 — 수축 사전확률(prior)이 낡았다: 고정상수 -> 시즌 인과 추세외삽 | `logs/results67__*` (2개) |
| [`run_dcn_v2_sota.py`](run_dcn_v2_sota.py) | Deep & Cross Network v2 (DCN-v2) | — |
| [`run_ft_pure.py`](run_ft_pure.py) | 트리 제거 FT-Transformer 순수 | — |
| [`run_ft_transformer_sota.py`](run_ft_transformer_sota.py) | Turing GPU Compatible FT-Transformer | — |
| [`run_ftt_clean.py`](run_ftt_clean.py) | FT-Transformer 정식 재측정 스크립트 | — |
| [`run_nn_pure.py`](run_nn_pure.py) | 트리 완전 제거 순수 NN 파이프라인 | — |
| [`run_pitchformer_sota.py`](run_pitchformer_sota.py) | 차세대 최신 딥러닝 아키텍처 [PitchFormer-SwiGLU] 구현 및 실측 | — |

---

## 특히 중요한 것

| 실험 | 결론 |
|---|---|
| `run29_cmp_retune.py` | 성분모델 재튜닝 **기각**. 성분 예측이 좋아지면 y 가 나빠진다 |
| `run46_b3_confirm.py` | `pbc_*` 채택 근거. 유일하게 LB 를 올린 피처 (+3.30) |
| `run56_blend_verify.py` | v17 블렌드 검증. CV 는 좋았으나 **LB -5.84** |
| `run57_fold2022_contam.py` | F리그 오염 검사. 현행 마스크가 옳다 |
| `run58_platoon.py` / `run59_pbc_rehier.py` | 투수x타자손 축. 잔차 202.9 는 실재하나 전환 실패 |
| `run63c_hier.py` | 계층 베이즈(from scratch). 단독 407/488 |
| `run64_fclean.py` | **1군/2군 오염 보정.** 폴드2024 +5.8 — 현재 유일 생존 후보 |
| `run67_prior.py` | 수축 사전확률이 낡았다는 실태 발견 (수정 시도는 실패) |
| `run_nn_pure.py` / `run_ftt_clean.py` | 순수 NN. 조기중단 홀드아웃 누수 문제 → `docs/TREE_VS_DL.md` |

## 재현

로컬은 `python run57_fold2022_contam.py` 로 바로 돈다 (`/home/lee/lga` 기준).
코랩은 [`../colab/05_hpo.ipynb`](../colab/05_hpo.ipynb) 또는 `03_run_experiments.ipynb` 를 써라.

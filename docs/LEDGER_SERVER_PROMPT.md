# 원장 수집·집계 서버 추가 프롬프트

이미 돌고 있는 `lgadata.hyunku.mmv.kr` 데이터 서버에 **업로드 + 집계** 기능을 붙인다.
아래를 홈서버 작업용 Claude Code 세션에 그대로 붙여넣으면 된다.

---

## 붙여넣을 프롬프트

전에 만든 데이터 서버(`lgadata.hyunku.mmv.kr`, nginx:alpine + Basic Auth + Cloudflare Tunnel)에
**두 가지를 추가**하려고 해.

### 1. 원장 업로드 받기 (WebDAV PUT)

팀원들이 Google Colab 에서 실험을 돌리면 `ledger_<이름>.csv` 가 생긴다.
이걸 서버로 올려서 한곳에 모으고 싶어.

- 경로: `/ledgers/` 아래에 PUT 으로 업로드
- Colab 에서 이렇게 올릴 수 있어야 한다:
  ```bash
  curl -u team:비번 -T ledger_윤제용.csv https://lgadata.hyunku.mmv.kr/ledgers/ledger_윤제용.csv
  ```
- nginx 의 `dav_methods PUT DELETE` 로 되는지 먼저 확인해줘 (alpine 이미지에 dav 모듈이 있는지)
- 없으면 아주 작은 업로드 엔드포인트를 따로 올려도 된다. **간단한 쪽으로 판단해줘**
- 기존 `/` 의 데이터 파일들은 **읽기 전용**으로 유지해야 한다. 업로드는 `/ledgers/` 에만 허용
- 같은 이름으로 다시 올리면 덮어쓰기 (팀원이 실험을 추가하면 파일이 계속 커진다)
- 인증은 기존 Basic Auth 그대로

### 2. 집계 페이지 자동 생성

`/ledgers/` 의 CSV 들을 합쳐서 **HTML 요약 페이지**를 만들어 `/summary.html` 로 서빙해줘.
5분마다(또는 업로드 시) 갱신되면 된다. cron 이든 뭐든 편한 방식으로.

CSV 컬럼은 이렇게 생겼다 (헤더 있음, UTF-8 BOM):

```
timestamp,runner,name,kind,params_json,m24,m23,m24_sd,m23_sd,delta24,delta23,
se24,se23,both,verdict,lb_lo,lb_hi,seeds,nfeat,sec,notes
```

**집계 로직은 단순하다. 판정은 이미 CSV 안에 계산돼 있으니 다시 계산하지 마라.**

페이지에 넣을 것:

- 맨 위 헤드라인: 총 건수, 실행자 수, **`verdict` 별 집계**
  (`verdict` 값은 `채택후보` / `보류` / `노이즈` / `기각` 넷 중 하나)
- `채택후보` 를 `delta24` 내림차순으로 정렬한 표 (제일 중요하다. 맨 위에)
- 나머지 판정도 접이식(`<details>`)으로
- 표 컬럼: 실행자, 아이디어(`name`), 설정(`params_json`), 폴드2024 효과(`delta24`),
  폴드2023 효과(`delta23`), LB 기대(`lb_lo`~`lb_hi`), 시드(`seeds`), 시각(`timestamp`)
- 실행자별 기여 건수
- 마지막 갱신 시각

한국어로. 표만 잘 보이면 되니까 CSS 는 최소로. 다크모드까지는 필요 없어.

### 주의

- **원장 파일이 하나도 없어도 페이지가 깨지지 않아야 한다** (팀이 아직 실험을 안 돌렸을 수 있다)
- CSV 가 깨져 있거나 컬럼이 모자라도 그 파일만 건너뛰고 나머지로 진행
- `verdict` 에 예상 못 한 값이 들어와도 죽지 말 것

### 산출물

- 변경된 `docker-compose.yml` / `nginx.conf`
- 집계 스크립트와 스케줄 설정
- **Colab 에서 쓸 업로드 명령 한 줄** (인증 포함, 완성된 형태로)
- 업로드가 실제로 되고 `/summary.html` 이 뜨는지 **직접 테스트해서 확인**해줘.
  테스트용 더미 CSV 를 만들어 올렸다가 지우면 된다

---

## 이쪽에서 할 것

서버가 준비되면 `src/sync.py` 의 업로드 경로를 git 에서 이 서버로 바꾼다.
현재는 git 기반으로 되어 있고, 서버 방식이 뜨면 그쪽을 기본으로 쓴다.

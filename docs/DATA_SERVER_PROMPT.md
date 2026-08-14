# 데이터 서버 구축 프롬프트

아래 내용을 **홈서버 작업용 Claude Code 세션에 그대로 붙여넣으면** 된다.

---

## 붙여넣을 프롬프트

집 서버에 도커로 **정적 파일 다운로드 서버**를 올리려고 해. 도메인은 `data.hyunku.mmv.kr` 이고,
AI 해커톤 팀원들이 **Google Colab 노트북에서 `wget`/`curl` 로 받아가는** 용도야.

### 서빙할 파일 (총 약 1.1GB)

```
대회 원본 데이터 (재배포 제한 있음 — 아래 접근제어 참고)
  train.csv               352M
  trackman_history.csv    338M
  test.csv                1.9K
  sample_submission.csv   112B

파생 캐시 (팀원들이 매번 재생성하면 30~60분 걸려서 미리 제공)
  X98.parquet             175M
  features.parquet        174M
  aligned.parquet          52M
  tm5.parquet             4.7M
  oof_comp.parquet         30M
  pitcher_map.csv          20K
  v6_tmsel.json           106B
```

### 요구사항

**1. 접근 제어 (중요)**
원본 데이터는 데이콘 대회 제공 데이터라 **공개 인터넷에 그대로 열면 안 된다.**
- 최소한 토큰/비밀번호 기반 접근 제어를 걸어줘. Nginx basic auth 또는 URL 토큰 방식 중 Colab에서 쓰기 편한 쪽으로 제안해줘
- 팀원 5명이 공유하는 단일 토큰이면 충분해. 사용자별 계정 관리는 과하다
- 디렉터리 목록(autoindex)은 인증 뒤에서만 보이게

**2. 대용량 다운로드 대응**
- **Range 요청(이어받기) 필수.** 352MB 파일을 Colab에서 받다가 끊기면 처음부터 다시 받는 건 곤란해
- `Content-Length` 정확히 내려줘야 진행률 표시가 된다
- 타임아웃을 넉넉히 (파일당 최소 10분)

**3. 무결성 검증**
- 각 파일의 **SHA256 체크섬을 담은 `MANIFEST.txt`** 를 같이 서빙해줘
- 팀원이 받은 뒤 검증할 수 있게, 그 파일에 `파일명  크기  sha256` 형식으로

**4. HTTPS**
- `data.hyunku.mmv.kr` 로 접속 가능해야 하고 TLS 필요
- 홈서버라 공인 IP/포트포워딩 상황을 먼저 물어봐줘. 상황에 따라 **Cloudflare Tunnel** 이 포트 개방 없이 깔끔할 수 있어 (그쪽이면 TLS도 자동)
- Let's Encrypt 직접 발급 방식과 Cloudflare Tunnel 방식의 장단점을 비교해서 추천해줘

**5. 도커 구성**
- `docker-compose.yml` 한 벌로 올라가게
- 데이터는 볼륨 마운트 (이미지에 굽지 마)
- 재부팅 후 자동 시작 (`restart: unless-stopped`)
- 컨테이너는 가벼운 걸로 (nginx-alpine 정도)

**6. 산출물**
- `docker-compose.yml`, nginx 설정, 배포 순서를 담은 `README.md`
- **Colab에서 쓸 다운로드 스니펫**을 완성된 형태로 (토큰 넣는 자리 표시, 이어받기 옵션 포함, 체크섬 검증까지)
- 데이터 파일을 서버 어느 경로에 두면 되는지 명확히

### 참고

- 서버 사양이나 OS는 네가 먼저 확인해줘
- 대역폭이 가정용이라 동시 5명이 350MB씩 받으면 느릴 수 있어. 그 부분 현실적인 예상 시간도 알려줘
- 나중에 파일이 추가될 수 있으니 폴더에 넣기만 하면 서빙되는 구조로

---

## 우리 쪽에서 준비해둘 것

이 서버가 뜨면 `src/config.py` 의 `DATA_URL` 을 채우고,
`colab/01_setup_and_data.ipynb` 의 다운로드 셀이 Drive 마운트 대신 이 서버를 쓰도록 전환한다.

체크섬 매니페스트 생성은 서버에 파일을 올릴 때 아래로 만들면 된다.

```bash
cd /path/to/served/files
{ printf "%-24s %12s  %s\n" "FILE" "SIZE" "SHA256"
  for f in *; do
    [ -f "$f" ] || continue
    printf "%-24s %12s  %s\n" "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
} > MANIFEST.txt
```

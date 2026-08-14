# 데이터 서버 구축 프롬프트

아래 내용을 **홈서버 작업용 Claude Code 세션에 그대로 붙여넣으면** 된다.

---

## 붙여넣을 프롬프트

집 서버에 도커로 **정적 파일 다운로드 서버**를 올리려고 해. 도메인은 `lgadata.hyunku.mmv.kr` 이고,
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
- **`MANIFEST.txt` 는 이미 만들어져 있다.** 파일들과 같은 폴더에 들어있고 `파일명 / 크기 / sha256` 형식이다.
  전송 전 원본에서 뜬 값이니 **새로 만들지 말고 그대로 서빙만** 해줘 (새로 뜨면 전송 중 손상을 못 잡는다)
- 팀원이 받은 뒤 이렇게 검증할 수 있어야 한다:
  ```bash
  sha256sum -c <(awk 'NR>2 {print $3"  "$1}' MANIFEST.txt)
  ```

**4. HTTPS**
- `lgadata.hyunku.mmv.kr` 로 접속 가능해야 하고 TLS 필요
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

## 파일을 홈서버로 옮기는 법

작업 PC(WSL2)의 `/home/lee/lga/_serve/` 에 11개 파일 + `MANIFEST.txt` 가 이미 모여 있다.
하드링크로 모은 것이라 원본과 같은 실체이며 디스크를 추가로 쓰지 않는다.

**같은 LAN + SSH (권장)**
```bash
rsync -avP --checksum /home/lee/lga/_serve/ 사용자@서버IP:/srv/lga-data/
```
`-P` 가 진행률 표시 + 이어받기. 끊겨도 재실행하면 이어진다.
기가비트 유선 20~40초, WiFi 2~5분.

WSL2 는 NAT 뒤에 있지만 **바깥으로 나가는 연결은 정상**이라 홈서버 IP 로 바로 붙는다.
반대 방향(서버에서 WSL 로 당기기)은 포트 프록시가 필요해 번거롭다. 미는 쪽이 낫다.

**외장 디스크**
```bash
cp -av /home/lee/lga/_serve/. /mnt/e/lga-data/    # 윈도우 E: 드라이브 예시
```

> **주의 — 경로 착각으로 한 번 크게 헤맸다.**
> 서버가 Proxmox 컨테이너(CT) 안에서 돌면, 서빙 경로(`/srv/lga-data`)는
> **컨테이너 안에만 존재**하고 호스트에는 없다. 호스트의 실제 위치는
> bind mount 원본(예: `/mnt/storage/lga-data`)이다.
> SSH 는 호스트로 붙기 때문에 컨테이너 경로로 rsync 하면
> **호스트 루트에 엉뚱한 새 디렉터리가 생기고 파일이 거기 고립된다.**
> 서버는 200 을 주는데 파일만 404 가 나는 상태가 된다.
> rsync 전에 `ssh 서버 'ls -la <경로>'` 로 **그 경로가 실제 서빙 위치인지** 먼저 확인해라.

**옮긴 뒤 서버에서 검증**
```bash
cd /srv/lga-data && sha256sum -c <(awk 'NR>2 {print $3"  "$1}' MANIFEST.txt)
```
전부 `OK` 여야 한다.

## 서버가 뜨면 우리 쪽에서 할 것

`src/config.py` 의 `DATA_URL` 을 채우고,
`colab/01_setup_and_data.ipynb` 의 다운로드 셀이 Drive 마운트 대신 이 서버를 쓰도록 전환한다.

# 과제 D-7 실습 가이드 — 공개 가능한 공간의 3D 뷰어 올리기

마감 **2026-11-20 (금) 23:59** · 제출: Pages 주소 한 줄 + 저장소 주소 한 줄

```
최종 주소   https://woochoi.github.io/three-dimension/4_homework/
장면 확인   https://woochoi.github.io/three-dimension/4_homework/scene.sog
저장소      https://github.com/woochoi/three-dimension
```

저장소 `three-dimension`의 루트가 로컬 `~/1_AI_AGENT_03/three-dimension/` 폴더이고, 과제는 그 안의 `4_homework/`에 둡니다. Pages를 main 브랜치 루트로 켜면 과제 주소는 위처럼 `/4_homework/`로 끝납니다.

## 최종 폴더 모양

```
4_homework/
├── index.html          뷰어 (three.js + Spark 2.4.0, 끌어다 놓기 비교 기능)
├── scene.sog           줄인 알갱이 파일  ← Pages에 올리는 장면
├── README.md           촬영·측정 결과 (이미 틀을 만들어 둠, 값을 채운다)
├── shots/viewer.png    화면 갈무리 한 장
├── .gitignore          영상·PLY 등 큰 파일 제외
├── scripts/            reconstruct.py · make_ply.py · convert.py (맥 로컬 실행)
├── .claude/skills/room-3d/   에이전트용 지침 (교육 자료의 데모 저장소에서 복사)
├── .venv/ · vendor/    맥 로컬 실행 환경 (올리지 않음)
└── raw/                내 컴퓨터에만 보관: 영상, 프레임, points.npz, scene.ply, 비교용 형식들
```

## 채점 기준 → 어디서 점수가 나오나

| 기준 (각 25점) | 해당 단계 |
|---|---|
| 공개 판단: 공개해도 되는 공간 + 근거 한 줄 | 1단계 → README §1 |
| 주소 동작: 다른 사람 폰에서 화면이 뜨고 파일이 내려옴 | 6·7단계 → README §5 |
| 수치 판정: 카메라 퍼짐, 형식별 크기, 알갱이 수, 알갱이당 바이트 | 3·4·5단계 → README §3·4 |
| 한계 설명: 안 나온 자리를 시차·정렬과 연결 | 7단계 → README §6 |

---

## 1단계. 공간 고르고 촬영하기 (직접, 30분)

**고르기** — 둘레를 걸을 수 있고 형태가 뚜렷한 곳: 공원 조형물, 정자, 계단, 동상 등.
- 촬영 정책 확인 (출입 가능 ≠ 촬영·공개 허용)
- 사람 얼굴, 차량 번호, 문패, 화면 속 개인정보가 **처음부터 안 들어오게** 찍는다
- 유리·물·금속 반사면, 햇빛에 하얗게 날아가는 면은 피한다

**찍기**
- **가로**, **15~20초**, 대상을 바라보며 **몸을 옆으로 옮겨 원호로** 천천히 걷는다 (제자리 회전 ✗)
- 앞뒤 프레임에 같은 부분이 이어서 보이게, 흔들려 번지지 않게
- 자세히 보여 주고 싶은 곳은 그 앞에서 좌우로 한두 걸음 더

**확인** — 다시 재생해 번짐·개인정보 확인 후 `4_homework/raw/` 에 복사 (예: `raw/space.mp4`).
README §1(공간, 공개 근거 한 줄)과 §2(걸은 경로)를 이때 바로 적어 둔다.

## 2단계. 로컬 환경 (맥북, 설치 완료)

Colab 대신 **이 맥북(Apple M5 Pro, 통합 메모리 25.8GB)의 GPU(MPS)** 로 복원합니다. 교육 자료는 T4를 쓰지만, 현재 MapAnything 코드는 `cuda → mps → cpu` 순으로 장치를 고르게 되어 있어 맥에서도 돕니다.

이미 준비해 둔 것:

| 항목 | 위치 |
|---|---|
| Python 3.12 가상환경 + torch 2.15 (MPS) | `4_homework/.venv/` |
| MapAnything 소스 (설치 경로를 `sys.path`에 반영) | `4_homework/vendor/map-anything/` |
| 모델 가중치 `facebook/map-anything-apache` | `~/.cache/huggingface/` (미리 받아 둠) |
| 복원·변환 스크립트 | `4_homework/scripts/` |

ffmpeg 없이 OpenCV로 프레임을 뽑으므로 따로 설치할 것이 없습니다. 환경을 처음부터 다시 만들려면:

```bash
cd ~/1_AI_AGENT_03/three-dimension/4_homework
git clone -q --depth 1 https://github.com/facebookresearch/map-anything.git vendor/map-anything
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision matplotlib -e vendor/map-anything
```

## 3단계. 영상 → 점 + 카메라 (맥북 MPS)

```bash
cd ~/1_AI_AGENT_03/three-dimension/4_homework
.venv/bin/python scripts/reconstruct.py raw/space.mp4
```

[scripts/reconstruct.py](scripts/reconstruct.py)는 교육 자료의 요청 사항을 그대로 구현했습니다: 프레임 24장 고르게 추출(긴 변 축소) → `fp16` → 각 축 0.5~99.5% 밖 점 제거 → 점·색·카메라를 `raw/points.npz` 하나에 저장 → 장치·복원 시간·GPU 메모리·점 개수·퍼짐 비 출력.

| 결과 파일 | 내용 |
|---|---|
| `raw/frames/` · `raw/frames.png` | 추출한 24장과 8장 미리보기 (번짐 확인) |
| `raw/cameras.png` | 위·정면에서 본 점 + 카메라 경로(빨간 선)와 바라보는 방향 |
| `raw/reconstruct.json` | README §3에 옮길 수치 |

**판정**
- 점 개수: 24장이면 300만 개 안팎
- 카메라 퍼짐 비 **0.1보다 커야** 함. 작으면 제자리 회전 → 재촬영 고려
- `cameras.png`의 빨간 경로가 내가 실제 걸은 원호와 닮았는지 영상 옆에 놓고 비교

→ README §3에 장치(맥 MPS)·복원 시간·메모리·점 개수·퍼짐 비·경로 비교를 적는다. **T4가 아니라 맥에서 돌렸다는 점도 적는다.**
메모리가 넘치면 `--frames 16`, fp16에서 결과가 이상하면 `--amp fp32` 로 다시.

> 막히면: 데모 `scene.sog`로 4~6단계를 먼저 진행하고, 막힌 위치·이유를 README에 적는다(인정됨). 데모 결과와 내 결과는 구분해 기록.

## 4단계. 점 → 가우시안 스플랫 PLY

```bash
.venv/bin/python scripts/make_ply.py              # K=1.4, 전체 점
.venv/bin/python scripts/make_ply.py --K 1.8      # 틈이 보이면 K↑, 뭉개지면 K↓
.venv/bin/python scripts/make_ply.py --n-keep 1500000   # 돌릴 때 끊기면 점 수↓
```

[scripts/make_ply.py](scripts/make_ply.py): INRIA 형식 이진 PLY, y·z 부호 뒤집기, 색 → 0차 구면 조화(`f_dc`), `f_rest` 없음, 반지름 `K × 긴 변 ÷ √N`, 크기는 로그, 불투명도 0.9는 로짓, 회전 (1,0,0,0), 법선 0, `N_KEEP`(`--n-keep`). 결과 `raw/scene.ply` + `raw/scene.json`(알갱이 수·반지름·긴 변·크기·저장 시간).

**판정**: 출력의 `알갱이당 68바이트`. 크게 다르면 헤더에 `f_rest`가 들어갔는지 확인.

## 5단계. 세 형식으로 변환·비교

```bash
.venv/bin/python scripts/convert.py
```

[scripts/convert.py](scripts/convert.py)가 `splat-transform@3.10.3 -w` 로 `.compressed.ply` · `.sog` · `.spz`를 만들고, 각 결과를 다시 PLY로 풀어 **알갱이 수가 같은지 확인**한 뒤 크기(바이트)·변환 시간·알갱이당 바이트 표를 `raw/formats.md`에 씁니다. 그 표를 README §4에 붙여 넣으면 됩니다.

> **3.10.3 고정 이유**: 3.11.0부터 기본인 SOG v3는 Spark 2.4.0이 못 읽는다.

**판정**: 세 결과의 알갱이 수가 같은지, `.sog`가 100MiB 아래인지. 크기만 보고 고르지 말고 뷰어에 끌어다 놓아 모서리·작은 물건을 같은 시점에서 비교.
→ 고른 파일만 `cp raw/scene.sog scene.sog` 로 `4_homework/`에 둔다.

## 6단계. 뷰어 확인 (`index.html` 준비 완료)

[index.html](index.html)은 교육 자료 요구 사항대로 만들어 두었고, 데모 `scene.sog`(알갱이 3,197,944개)로 열리는 것까지 확인했습니다.
- import map: three 0.180.0, Spark 2.4.0 (`sparkjs.dev/releases`), 빌드 없음
- 파일 이름은 맨 위 `SPLAT_FILE` 한 줄 (주소 뒤 `?f=다른파일` 로도 바꿀 수 있음)
- SparkRenderer · OrbitControls(감쇠) · Spark `getBoundingBox` 로 카메라 배치 · 위아래 뒤집기
- 진행 상태 5단계 + 파일 크기 · 받은 시간 · 알갱이 수 · 푸는 시간 표시
- **파일 끌어다 놓기 / 파일 열기**로 `.ply` · `.compressed.ply` · `.sog` · `.spz` 비교 → 오른쪽 위 **비교 표**에 바이트 · 시간 · 알갱이 수 · 알갱이당 바이트가 쌓임

**로컬 확인**
```bash
cd ~/1_AI_AGENT_03/three-dimension/4_homework && python3 -m http.server 8777
```
`http://localhost:8777` — 뒤집힘(천장이 아래 → y·z 부호), 검은 화면(SparkRenderer·pcsogszip), 틈(K↑)·뭉개짐(K↓) 확인.
화면을 갈무리해 `shots/viewer.png`로 저장.

## 7단계. GitHub Pages 공개 (15분)

**올리기 전 마지막 점검** — 화면 속 얼굴·번호판·문패 없음, `raw/`·`.ply`가 `.gitignore`로 빠졌는지.

```bash
cd ~/1_AI_AGENT_03/three-dimension
touch .nojekyll
git init -b main
git remote add origin https://github.com/woochoi/three-dimension.git
git add .nojekyll 4_homework
git status            # raw/, *.ply, 영상이 목록에 없어야 함
git commit -m "D-7: 공개 가능한 공간의 3D 뷰어"
git push -u origin main
```

Pages 켜기 (Settings → Pages → Source: Deploy from a branch → `main` / `(root)`), 또는:
```bash
gh api -X POST repos/woochoi/three-dimension/pages -f "source[branch]=main" -f "source[path]=/"
```

1~2분 뒤 확인:
```bash
curl -sI https://woochoi.github.io/three-dimension/4_homework/ | head -1
curl -sI https://woochoi.github.io/three-dimension/4_homework/scene.sog | head -3
```

**폰 측정** — 내 폰, 그리고 **다른 사람 폰**(첫 방문)으로 주소를 열고 첫 화면까지 시간, 기기, 통신 환경을 README §5에 기록. 돌려 보며 구멍·얼룩이 드러나는 방향을 원본 영상과 대조해 README §6에 기록 → 다시 커밋·푸시.

## 8단계. 제출

- Pages 주소: `https://woochoi.github.io/three-dimension/4_homework/`
- 저장소 주소: `https://github.com/woochoi/three-dimension` (README 직접 링크: `.../tree/main/4_homework`)
- 보내기 전 `/scene.sog`가 내려오는지 한 번 더. 채점 전까지 재제출 가능.
- 원본 영상·프레임·점 파일·PLY는 `raw/`에 보관.

## 체크리스트

- [ ] 1 공간 선정 · 공개 근거 한 줄 · 촬영 (가로, 15~20초, 원호)
- [x] 2 맥 로컬 환경 (.venv + MapAnything + 모델 가중치)
- [ ] 3 점 개수 · 퍼짐 비 > 0.1 · 결과 내려받기
- [ ] 4 `scene.ply` · 알갱이당 ≈68바이트
- [ ] 5 세 형식 변환 · 크기/시간/개수 표
- [ ] 6 `index.html` 로컬에서 열림 · `shots/viewer.png`
- [ ] 7 Pages 공개 · `/scene.sog` 내려옴 · 두 폰 측정
- [ ] 8 README 모든 칸 채움 · 제출

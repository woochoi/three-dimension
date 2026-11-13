# 로컬 실행 환경과 모델 설치 위치

과제 D-7을 Colab 대신 **맥북(Apple M5 Pro, 통합 메모리 25.8GB)의 GPU(MPS)** 에서 진행하면서 설치한 것들의 위치와 용량입니다. 측정일 2026-11-13.

이번 작업으로 생긴 용량은 모두 합쳐 **약 6.7GB**입니다.

## 1. 모델 가중치: 4.9GB (홈 폴더의 Hugging Face 캐시)

모델은 프로젝트 폴더가 아니라 홈 폴더의 캐시에 받아졌습니다. 처음 `MapAnything.from_pretrained("facebook/map-anything-apache")`를 실행할 때 자동으로 내려받습니다.

| 항목 | 위치 | 용량 |
|---|---|---|
| `facebook/map-anything-apache` 가중치 (`model.safetensors`) | `~/.cache/huggingface/hub/blobs/3b/3b0c1c38…` | **4.91GB** |
| 모델 폴더 (설정 파일과 링크만) | `~/.cache/huggingface/hub/models--facebook--map-anything-apache/` | 24KB |
| DINOv2 코드 (모델이 내부에서 불러옴) | `~/.cache/torch/hub/facebookresearch_dinov2_main/` | 4.3MB |

- 모델 폴더가 24KB밖에 안 되는 이유: 실제 가중치 파일은 공용 저장소 `hub/blobs/`에 있고, 모델 폴더의 `snapshots/…/model.safetensors`는 그 파일을 가리키는 링크입니다.
- `~/.cache/huggingface` 전체는 8.0GB입니다. 나머지 약 3GB는 예전부터 있던 다른 모델(chatterbox, faster-whisper)로 이번 작업과 관계없습니다.
- `~/.cache/torch/hub/checkpoints/`의 84MB 파일(2026-10-30 생성)도 이번 작업과 관계없습니다.

## 2. 프로젝트 폴더 안 (`4_homework/`)

| 항목 | 용량 | 내용 |
|---|---|---|
| `.venv/` | 1.4GB | Python 3.12 가상환경 + torch 2.15 (MPS) 등 |
| `vendor/map-anything/` | 18MB | MapAnything 소스 코드 (`sys.path`에 넣어 사용) |
| `raw/` | 346MB | 원본 영상 `cup.mp4`, 프레임 24장, 점 파일, PLY, 형식별 변환 파일 |

이 세 폴더와 모델 캐시는 모두 `.gitignore`로 빠져 있어 GitHub에는 올라가지 않습니다.

## 3. 환경을 처음부터 다시 만들기

```bash
cd ~/1_AI_AGENT_03/three-dimension/4_homework
git clone -q --depth 1 https://github.com/facebookresearch/map-anything.git vendor/map-anything
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision matplotlib -e vendor/map-anything
```

모델 가중치(4.9GB)는 첫 복원 실행 때 자동으로 받아지고, 미리 받아 두려면:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/map-anything-apache')"
```

## 4. 공간을 비우고 싶을 때

과제가 끝난 뒤 지워도 됩니다. 다시 복원하려면 가중치 4.9GB를 새로 받아야 합니다.

```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--map-anything-apache ~/.cache/huggingface/hub/blobs/3b/3b0c1c38*
```

```bash
rm -rf ~/1_AI_AGENT_03/three-dimension/4_homework/.venv ~/1_AI_AGENT_03/three-dimension/4_homework/vendor
```

`raw/`에는 원본 영상과 PLY가 있으니 제출과 채점이 끝날 때까지는 남겨 둡니다. 형식을 다시 비교하거나 알갱이 크기 K를 바꿔 볼 때 필요합니다.

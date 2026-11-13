"""영상 → 프레임 24장 → MapAnything (맥 MPS) → 점·색·카메라 저장.

사용:  .venv/bin/python scripts/reconstruct.py raw/space.mp4 [--frames 24] [--long-side 1024]
결과:  raw/frames/*.jpg, raw/frames.png, raw/points.npz, raw/cameras.png, raw/reconstruct.json
"""
import argparse, json, os, sys, time, platform
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "map-anything"))  # 설치 경로를 모듈 검색 경로에 반영

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--long-side", type=int, default=1024, help="추출 프레임 긴 변(px). 모델은 내부에서 518 기준으로 다시 줄임")
ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "fp32"])
ap.add_argument("--out", default=str(ROOT / "raw"))
a = ap.parse_args()

out = Path(a.out); fdir = out / "frames"; fdir.mkdir(parents=True, exist_ok=True)
for f in fdir.glob("*.jpg"):
    f.unlink()
stats = {}

# ---------- 1. 영상 정보 + 프레임 고르게 추출 ----------
t = time.time()
cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur = n / fps if fps else 0
stats["video"] = {"file": os.path.basename(a.video), "width": w, "height": h, "fps": round(fps, 2),
                  "frames": n, "duration_s": round(dur, 2)}
print(f"영상 {w}x{h}, {fps:.1f}fps, {dur:.1f}초, 프레임 {n}")
if h > w:
    print("  ! 세로 영상입니다. 가로가 낫습니다 (천장·바닥이 화면을 차지).")
if not 12 <= dur <= 25:
    print("  ! 15~20초 권장 범위를 벗어났습니다.")

# 마지막 프레임은 탐색이 실패하는 경우가 있어 살짝 앞에서 끝낸다
idx = np.linspace(0, max(n - 3, 1), a.frames).round().astype(int)
paths = []
for i, k in enumerate(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, img = cap.read()
    if not ok:
        continue
    s = a.long_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p = fdir / f"f_{i:03d}.jpg"; cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95]); paths.append(str(p))
cap.release()
stats["frames_extracted"] = len(paths)
print(f"프레임 {len(paths)}장 추출 {time.time()-t:.1f}초 → {fdir}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(16, 5))
for axx, p in zip(ax.ravel(), paths[:: max(1, len(paths) // 8)][:8]):
    axx.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)); axx.axis("off"); axx.set_title(Path(p).name, fontsize=8)
plt.tight_layout(); plt.savefig(out / "frames.png", dpi=80); plt.close()

# ---------- 2. 모델 ----------
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

dev = "mps" if torch.backends.mps.is_available() else "cpu"
stats["device"] = f"{dev} / {platform.processor() or platform.machine()} / torch {torch.__version__}"
print("장치:", stats["device"])
t = time.time()
model = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
stats["model_load_s"] = round(time.time() - t, 1); print(f"모델 준비 {stats['model_load_s']}초")

views = load_images(paths)
print(f"모델 입력 크기 {tuple(views[0]['img'].shape[-2:])}")
t = time.time()
with torch.no_grad():
    pred = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                       amp_dtype=a.amp, apply_mask=True, mask_edges=True, apply_confidence_mask=False)
if dev == "mps":
    torch.mps.synchronize()
stats["reconstruct_s"] = round(time.time() - t, 1)
if dev == "mps":
    stats["gpu_mem_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
print(f"복원 {stats['reconstruct_s']}초, GPU 메모리(드라이버 할당) {stats.get('gpu_mem_driver_gb')}GB")

# ---------- 3. 점 모으기 + 튄 점 제거 + 퍼짐 비 ----------
pts, cols, cams = [], [], []
for v in pred:
    m = v["mask"][0, ..., 0].cpu().numpy().astype(bool)
    pts.append(v["pts3d"][0].float().cpu().numpy()[m])
    cols.append((v["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[m])
    cams.append(v["camera_poses"][0].float().cpu().numpy())
xyz = np.concatenate(pts); rgb = np.concatenate(cols); cams = np.stack(cams)
stats["points_raw"] = int(len(xyz))
lo = np.percentile(xyz, 0.5, axis=0); hi = np.percentile(xyz, 99.5, axis=0)
keep = np.all((xyz > lo) & (xyz < hi), axis=1); xyz, rgb = xyz[keep], rgb[keep]
stats["points"] = int(len(xyz))
cpos = cams[:, :3, 3]
spread = float(np.linalg.norm(cpos.max(0) - cpos.min(0))); diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
stats.update(camera_spread=round(spread, 3), scene_diag=round(diag, 3), spread_ratio=round(spread / diag, 4))
print(f"점 {stats['points_raw']:,} → 튄 점 제거 후 {stats['points']:,}")
print(f"카메라 퍼짐 {spread:.3f} / 장면 대각선 {diag:.3f} = {spread/diag:.4f}",
      "(걸으며 찍은 것으로 보임)" if spread / diag > 0.1 else "(! 카메라가 거의 안 움직임 → 재촬영 고려)")

np.savez(out / "points.npz", xyz=xyz.astype(np.float32), rgb=rgb, cams=cams)

# ---------- 4. 카메라 배치 그림 (위에서 본 x-z, 옆에서 본 x-y) ----------
sub = np.random.default_rng(0).choice(len(xyz), min(60000, len(xyz)), replace=False)
fwd = cams[:, :3, 2]  # 카메라가 바라보는 방향 (OpenCV 규약 +z)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for axx, (i, j, name) in zip(ax, [(0, 2, "top view (x-z)"), (0, 1, "front view (x-y, y down)")]):
    axx.scatter(xyz[sub, i], xyz[sub, j], c=rgb[sub] / 255, s=0.3)
    axx.plot(cpos[:, i], cpos[:, j], "-o", c="red", ms=3, lw=1)
    L = diag * 0.05
    axx.quiver(cpos[:, i], cpos[:, j], fwd[:, i] * L, fwd[:, j] * L, color="red", angles="xy", scale_units="xy", scale=1)
    axx.text(cpos[0, i], cpos[0, j], " start", color="red"); axx.set_aspect("equal"); axx.set_title(name)
ax[1].invert_yaxis()
plt.tight_layout(); plt.savefig(out / "cameras.png", dpi=90); plt.close()

(out / "reconstruct.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(f"저장: {out/'points.npz'}, {out/'cameras.png'}, {out/'reconstruct.json'}")

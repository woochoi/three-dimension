"""points.npz → INRIA 형식 가우시안 스플랫 PLY (17 속성, 알갱이당 68바이트).

사용:  .venv/bin/python scripts/make_ply.py [--K 1.4] [--n-keep 0] [--in raw/points.npz] [--out raw/scene.ply]
"""
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=str(ROOT / "raw" / "points.npz"))
ap.add_argument("--out", default=str(ROOT / "raw" / "scene.ply"))
ap.add_argument("--K", type=float, default=1.4, help="구멍이 보이면 올리고, 뭉개지면 내린다")
ap.add_argument("--n-keep", type=int, default=0, help="0이면 전체, 숫자면 그만큼 고르게 뽑기")
ap.add_argument("--opacity", type=float, default=0.9)
a = ap.parse_args()

SH_C0 = 0.28209479177387814
d = np.load(a.inp); xyz, rgb = d["xyz"], d["rgb"]
if a.n_keep and a.n_keep < len(xyz):
    sel = np.random.default_rng(0).choice(len(xyz), a.n_keep, replace=False); sel.sort()
    xyz, rgb = xyz[sel], rgb[sel]

t = time.time()
n = len(xyz)
extent = float((xyz.max(0) - xyz.min(0)).max())
r = a.K * extent / np.sqrt(n)

p = xyz.astype(np.float32).copy()
p[:, 1] *= -1.0; p[:, 2] *= -1.0          # 모델은 y 아래, three.js 는 y 위 → y, z 부호 뒤집기

out = np.zeros((n, 17), dtype=np.float32)
out[:, 0:3] = p                                             # 위치
# 3:6 법선 = 0
out[:, 6:9] = (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0  # 기본 색 (0차 구면 조화)
out[:, 9] = np.log(a.opacity / (1.0 - a.opacity))           # 불투명도 (시그모이드 전 값)
out[:, 10:13] = np.log(r)                                    # 크기 (로그)
out[:, 13] = 1.0                                             # 안 돌린 사원수 (1,0,0,0)

props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
hdr = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
       + "".join(f"property float {k}\n" for k in props) + "end_header\n").encode()
with open(a.out, "wb") as f:
    f.write(hdr); f.write(out.tobytes())
dt = time.time() - t
size = Path(a.out).stat().st_size
stats = dict(splats=n, K=a.K, n_keep=a.n_keep, radius=round(r, 6), longest_side=round(extent, 4),
             ply_bytes=size, header_bytes=len(hdr), bytes_per_splat=round((size - len(hdr)) / n, 2),
             bytes_per_splat_incl_header=round(size / n, 3), save_s=round(dt, 2))
print(f"알갱이 {n:,}개, 반지름 {r:.5f} (장면 긴 변 {extent:.3f}, K={a.K})")
print(f"{a.out}: {size:,} 바이트 ({size/1e6:.1f}MB), 알갱이당 {stats['bytes_per_splat']}바이트, 저장 {dt:.2f}초")
Path(a.out).with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

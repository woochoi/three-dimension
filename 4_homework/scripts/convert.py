"""scene.ply → .compressed.ply / .sog / .spz 변환·비교 표 (splat-transform 3.10.3 고정).

알갱이 수는 각 결과를 다시 PLY로 풀어 헤더에서 읽어 확인한다.
사용:  .venv/bin/python scripts/convert.py [raw/scene.ply]
결과:  raw/scene.compressed.ply, raw/scene.sog, raw/scene.spz, raw/formats.md, raw/formats.json
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "raw" / "scene.ply")
ST = ["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w"]


def ply_count(p: Path) -> int:
    with open(p, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    raise ValueError(p)


def decoded_count(p: Path) -> int:
    with tempfile.TemporaryDirectory() as td:
        back = Path(td) / "back.ply"
        subprocess.run(ST + [str(p), str(back)], check=True, capture_output=True)
        return ply_count(back)


rows = [{"format": ".ply (원본)", "file": src.name, "bytes": src.stat().st_size, "convert_s": None, "splats": ply_count(src)}]
stem = src.name[:-4]
for ext in [".compressed.ply", ".sog", ".spz"]:
    dst = src.with_name(stem + ext)
    t = time.time()
    subprocess.run(ST + [str(src), str(dst)], check=True)
    dt = time.time() - t
    rows.append({"format": ext, "file": dst.name, "bytes": dst.stat().st_size, "convert_s": round(dt, 1),
                 "splats": decoded_count(dst)})

for r in rows:
    r["MB"] = round(r["bytes"] / 1e6, 2)
    r["bytes_per_splat"] = round(r["bytes"] / r["splats"], 2)
    r["ratio_vs_ply"] = round(rows[0]["bytes"] / r["bytes"], 1)

lines = ["| 형식 | 파일 크기 (바이트) | MB | 변환 시간 | 알갱이 수 | 알갱이당 바이트 | PLY 대비 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    cs = "–" if r["convert_s"] is None else f"{r['convert_s']}초"
    lines.append(f"| `{r['format']}` | {r['bytes']:,} | {r['MB']} | {cs} | {r['splats']:,} | {r['bytes_per_splat']} | {r['ratio_vs_ply']}배 |")
table = "\n".join(lines)
print("\n" + table)
same = len({r["splats"] for r in rows}) == 1
print("\n알갱이 수 모두 같음" if same else "\n! 알갱이 수가 다름 — 확인 필요")
sog = next(r for r in rows if r["format"] == ".sog")
print(f".sog {sog['bytes']/2**20:.1f}MiB", "(100MiB 아래, GitHub 업로드 가능)" if sog["bytes"] < 100 * 2**20 else "(! 100MiB 초과)")
(src.parent / "formats.md").write_text(table + "\n")
(src.parent / "formats.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

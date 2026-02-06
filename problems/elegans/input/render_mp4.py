import json
import sys
from pathlib import Path

import numpy as np


def render_mp4(sim_json_path: str, out_mp4: str, fps: int = 30):
    try:
        import imageio.v3 as iio
    except Exception as e:
        raise RuntimeError("imageio is required to render mp4") from e

    data = json.loads(Path(sim_json_path).read_text())
    pos = np.asarray(data.get("positions", []), dtype=float)
    curv = np.asarray(data.get("curvature", []), dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions missing or invalid")
    if curv.ndim != 2:
        raise ValueError("curvature missing or invalid")

    t = min(len(pos), len(curv))
    pos = pos[:t]
    curv = curv[:t]
    segs = curv.shape[1]

    # Simple body render: centerline along heading with curvature offsets
    frames = []
    for i in range(t):
        # Build a simple polyline around the position
        base = pos[i]
        xs = np.linspace(-0.5, 0.5, segs)
        ys = curv[i] * 0.3
        pts = np.stack([base[0] + xs, base[1] + ys], axis=1)

        # Render to an image
        w, h = 512, 512
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Normalize to image coords
        min_xy = pos.min(axis=0)
        max_xy = pos.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        norm = (pts - min_xy) / span
        pix = (norm * np.array([w - 1, h - 1])).astype(int)
        pix[:, 1] = (h - 1) - pix[:, 1]

        for p in pix:
            x, y = p
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = [0, 0, 0]
        frames.append(img)

    iio.imwrite(out_mp4, frames, fps=fps)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: render_mp4.py sim_output.json out.mp4 [fps]")
        sys.exit(1)
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    render_mp4(sys.argv[1], sys.argv[2], fps=fps)

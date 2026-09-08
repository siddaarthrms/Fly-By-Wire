#!/usr/bin/env python3
"""Render circuit-lightup PNGs from a demo telemetry run folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib required: pip install matplotlib") from exc

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_run_dir(raw: Path) -> Path:
    """Accept repo-relative paths like demos/telemetry/<run> from any cwd."""
    p = Path(raw)
    if p.is_dir():
        return p.resolve()
    candidate = (REPO_ROOT / p).resolve()
    if candidate.is_dir():
        return candidate
    return p.resolve()


def _load_run(run_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    grids = np.load(run_dir / "grids.npy")
    topo = np.load(run_dir / "circuit_topology.npz")
    frames = np.load(run_dir / "circuit_frames.npz")
    return meta, grids, topo, dict(frames)


def _grid_channel_map(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    idx: np.ndarray,
    contrib: np.ndarray,
) -> np.ndarray:
    out = np.zeros((30, 30), dtype=np.float32)
    counts = np.zeros((30, 30), dtype=np.float32)
    for i, neuron_idx in enumerate(idx):
        gx = int(grid_x[neuron_idx])
        gy = int(grid_y[neuron_idx])
        out[gy, gx] += float(contrib[i])
        counts[gy, gx] += 1.0
    mask = counts > 0
    out[mask] /= counts[mask]
    return out


def render_frame(run_dir: Path, frame_idx: int, *, out_path: Path | None = None) -> Path:
    meta, grids, topo, frames = _load_run(run_dir)
    n_frames = int(grids.shape[0])
    idx = int(frame_idx)
    if idx < 0:
        idx = n_frames + idx
    if idx < 0 or idx >= n_frames:
        raise SystemExit(f"frame_idx out of range: {frame_idx} (run has {n_frames} frames)")

    grid = grids[idx]
    gx = topo["grid_x"]
    gy = topo["grid_y"]
    right_map = _grid_channel_map(gx, gy, topo["right_idx"], frames["right_contrib"][idx])
    left_map = _grid_channel_map(gx, gy, topo["left_idx"], frames["left_contrib"][idx])
    brake_map = _grid_channel_map(gx, gy, topo["brake_idx"], frames["brake_contrib"][idx])
    throttle_map = _grid_channel_map(gx, gy, topo["throttle_idx"], frames["throttle_contrib"][idx])

    if out_path is None:
        out_dir = run_dir / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"circuit_frame_{idx:05d}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    title = meta.get("label", run_dir.name)
    fig.suptitle(f"{title} — frame {idx}", fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], grid, "gray", "Retina (30x30)"),
        (axes[0, 1], right_map, "Purples", "RIGHT steer drive"),
        (axes[0, 2], left_map, "Blues", "LEFT steer drive"),
        (axes[1, 0], brake_map, "Reds", "Brake channel"),
        (axes[1, 1], throttle_map, "Greens", "Throttle channel"),
    ]
    for ax, data, cmap, label in panels:
        if cmap == "gray":
            ax.imshow(data, origin="upper", cmap="gray", vmin=0, vmax=255)
        else:
            vmax = float(np.max(data)) if np.any(data) else 1.0
            ax.imshow(data, origin="upper", cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 2].axis("off")
    summary_lines = [
        f"right_push={frames['right_contrib'][idx].sum():.3f}",
        f"left_push={frames['left_contrib'][idx].sum():.3f}",
        f"brake_syn_sum={frames['brake_contrib'][idx].sum():.3f}",
        f"throttle_syn_sum={frames['throttle_contrib'][idx].sum():.3f}",
    ]
    if (run_dir / "frames.csv").is_file():
        import csv

        with (run_dir / "frames.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if idx < len(rows):
            row = rows[idx]
            summary_lines.extend(
                [
                    f"steer={float(row.get('steer', 0)):+.3f}",
                    f"brake={float(row.get('brake', 0)):.2f}",
                    f"throttle={float(row.get('throttle', 0)):.2f}",
                    f"t={float(row.get('t_sec', 0)):.2f}s",
                ]
            )
    axes[1, 2].text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8fafc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Telemetry folder from demos/telemetry/<label>_<time>/")
    ap.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    ap.add_argument(
        "--all-keyframes",
        type=int,
        default=0,
        metavar="N",
        help="If >0, also render N evenly spaced frames across the run",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    if not (run_dir / "meta.json").is_file():
        raise SystemExit(
            f"Not a telemetry run folder: {run_dir}\n"
            f"Expected meta.json under demos/telemetry/<label>_<timestamp>/"
        )

    out = render_frame(run_dir, args.frame, out_path=args.output)
    print(f"Wrote {out}")

    if int(args.all_keyframes) > 0:
        grids = np.load(run_dir / "grids.npy")
        n = int(grids.shape[0])
        picks = np.linspace(0, max(0, n - 1), num=int(args.all_keyframes), dtype=int)
        for pick in picks:
            p = render_frame(run_dir, int(pick))
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()

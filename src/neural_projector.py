import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "data" / "models" / "fly_pilot_production_v1.csv"
DEFAULT_PIXEL_MAP = REPO_ROOT / "data" / "mappings" / "fly_eye_grid_full.csv"


# Match your provided snippet IDs.
RIGHT_STEER_ID = 720575940633816986
LEFT_STEER_ID = 720575940625653029

# Offset-corrected right emphasis (see steering_bias below).
# R0 = mean(right_push) on straight-road calibration video; recompute if weights / grid change.
# With scale>1, left-turn clips gain negative mean bias without shifting straight mean (unlike naive r - w*l).
DEFAULT_R0_RIGHT_MEAN = 3.273860537592521
DEFAULT_RIGHT_PUSH_SCALE = 1.12


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural projector: video -> steering activation")
    parser.add_argument("--video", default="StraightDriving.mp4", help="Path to the input mp4 video")
    parser.add_argument("--max_frames", type=int, default=0, help="If >0, only process this many frames")
    parser.add_argument("--no-gui", action="store_true", help="Do not attempt to show cv2.imshow")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to steering weights CSV")
    parser.add_argument("--pixel_map", default=str(DEFAULT_PIXEL_MAP), help="Path to retinotopy grid CSV")
    parser.add_argument("--output_csv", default="steering_activation_timeseries.csv")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame prints")
    parser.add_argument(
        "--right-push-scale",
        type=float,
        default=DEFAULT_RIGHT_PUSH_SCALE,
        help=(
            "Emphasize right channel vs left with straight-road mean preserved: "
            "bias = scale*right - left - (scale-1)*R0. "
            "1.0 = legacy (right-left). Default %(default)s strengthens left-turn response; try 1.08–1.15."
        ),
    )
    parser.add_argument(
        "--r0-right-mean",
        type=float,
        default=DEFAULT_R0_RIGHT_MEAN,
        help=(
            "Reference mean right_push from straight calibration (used only when right-push-scale != 1). "
            "Recompute: neural_projector on straight clip, then mean(right_push) from CSV."
        ),
    )
    args = parser.parse_args()

    weights_df = pd.read_csv(args.weights)
    pixel_map = pd.read_csv(args.pixel_map)

    # Mapped input neurons (subset with coordinates)
    input_neuron_ids = pixel_map["pt_root_id"].astype("int64").values
    idx_map = {int(nid): i for i, nid in enumerate(input_neuron_ids)}

    # Precompute incoming edges to each steering target:
    # total_signal(target) = sum_incoming brightness(sender) * synapse_count(pre->target)
    def build_incoming(target_id: int):
        incoming = weights_df[weights_df["post_pt_root_id"] == target_id].copy()
        incoming = incoming[incoming["pre_pt_root_id"].isin(idx_map.keys())]
        if len(incoming) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        sender_ids = incoming["pre_pt_root_id"].astype("int64").values
        syn_counts = incoming["synapse_count"].astype("float64").values
        sender_idx = np.array([idx_map[int(sid)] for sid in sender_ids], dtype=np.int64)
        return sender_idx, syn_counts

    right_sender_idx, right_syn_counts = build_incoming(RIGHT_STEER_ID)
    left_sender_idx, left_syn_counts = build_incoming(LEFT_STEER_ID)

    # Pixel->grid lookups for all mapped neurons
    grid_x = pixel_map["grid_x"].astype("int64").values
    grid_y = pixel_map["grid_y"].astype("int64").values

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not args.quiet:
        print(f"Processing: {args.video} (frames={total_frames}, fps={fps:.2f})")
        print("Computing steering bias per frame...")

    # The "Physical Mass" Filter — rolling average over recent frames
    history = []
    WINDOW_SIZE = 5  # 5 frames at 30fps is ~160ms of "weight"

    records = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fly_vision = cv2.resize(gray, (30, 30), interpolation=cv2.INTER_AREA)

        # Brightness per mapped neuron: brightness is per neuron pixel location
        # grid brightness is 0..255; normalize to 0..1
        brightness = (fly_vision[grid_y, grid_x] / 255.0).astype("float64")

        # Visual "reflex": sum brightness(sender) * synapse_count(pre->target)
        right_push = float(np.sum(brightness[right_sender_idx] * right_syn_counts)) if len(right_sender_idx) else 0.0
        left_push = float(np.sum(brightness[left_sender_idx] * left_syn_counts)) if len(left_sender_idx) else 0.0
        w = float(args.right_push_scale)
        r0 = float(args.r0_right_mean)
        # Preserves mean bias on straight when right≈R0; boosts (r-l) when r>R0, pulls negative when r<R0 (typ. left-turn clip).
        steering_bias = w * right_push - left_push - (w - 1.0) * r0
        history.append(steering_bias)
        if len(history) > WINDOW_SIZE:
            history.pop(0)
        smoothed_steer = sum(history) / len(history)

        if not args.quiet:
            print(
                f"Biological RAW: {steering_bias:.4f} | SMOOTHED: {smoothed_steer:.4f}",
                flush=True,
            )

        records.append(
            {
                "frame_idx": frame_idx,
                "right_push": right_push,
                "left_push": left_push,
                "steering_bias": steering_bias,
                "smoothed_steer": smoothed_steer,
            }
        )

        if not args.no_gui:
            cv2.putText(
                frame,
                f"RAW: {steering_bias:.4f} | SMOOTH: {smoothed_steer:.4f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("FlyVision Output", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output_csv, index=False)

    # Report sanity stats for "straight road"
    abs_bias = np.abs(df["steering_bias"].values)
    abs_smooth = np.abs(df["smoothed_steer"].values)
    if not args.quiet:
        print(f"Frames processed: {len(df)}")
        print(f"Steering bias mean: {df['steering_bias'].mean():.6f}")
        print(f"Steering bias mean(|bias|): {abs_bias.mean():.6f}")
        print(f"Steering bias max(|bias|): {abs_bias.max():.6f}")
        print(f"Smoothed mean(|smoothed_steer|): {abs_smooth.mean():.6f}")
        print(f"Smoothed max(|smoothed_steer|): {abs_smooth.max():.6f}")
        print("First 10 frame biases:", ", ".join(f"{x:.4f}" for x in df["steering_bias"].head(10).values))
        print(
            "First 10 smoothed:",
            ", ".join(f"{x:.4f}" for x in df["smoothed_steer"].head(10).values),
        )
        print(f"Saved: {args.output_csv}")
    else:
        print(
            f"[{args.output_csv}] frames={len(df)} "
            f"left_mean={df['left_push'].mean():.4f} right_mean={df['right_push'].mean():.4f} "
            f"bias_mean={df['steering_bias'].mean():.6f} mean|bias|={abs_bias.mean():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()


#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compute SARM progress values for RA-BC (Reward-Aware Behavior Cloning) weighting.

This script processes all frames in a dataset with SARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses multi-output extraction: each SARM query returns progress for 9 frames, so we only
need ~num_frames/30 queries instead of one per frame (~30x speedup).

Usage:
    # Full RA-BC computation with visualizations
    python src/lerobot/policies/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Faster computation with stride (compute every 5 frames, interpolate the rest)
    python src/lerobot/policies/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --stride 5

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/policies/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 5

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset

from .modeling_sarm import SARMRewardModel
from .processor_sarm import make_sarm_pre_post_processors
from .sarm_utils import normalize_stage_tau


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    """Read reward_model_path from parquet metadata if available."""
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def load_sarm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    device: str = "cuda",
) -> tuple[LeRobotDataset, SARMRewardModel, any]:
    """
    Load SARM model, dataset, and preprocessor.

    Returns:
        Tuple of (dataset, reward_model, preprocessor)
    """
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices

    logging.info(f"Loading dataset: {dataset_repo_id}")
    temp_dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    fps = temp_dataset.fps

    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    return dataset, reward_model, preprocess


def to_numpy_image(img) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        # Take center frame for bidirectional sampling
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # Handle normalized images (may have negative values or values > 1)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames.

    Same as sarm_inference_visualization.py
    """
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # Progress plot
    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    # Sample frames
    ax_frames.axis("off")
    num_sample = 8
    sample_indices = np.linspace(0, len(frames) - 1, num_sample, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[idx])][:12]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_episode_prediction_mp4(
    dataset: LeRobotDataset,
    image_key: str,
    ep_start: int,
    ep_end: int,
    progress_preds: np.ndarray,
    stage_preds: np.ndarray,
    stage_labels: list[str],
    output_path: Path,
    title: str,
    fps: int = 20,
    gt_progress: np.ndarray | None = None,
):
    """Save per-frame prediction dynamics as an MP4 video."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("MP4 export requires opencv-python(-headless). Please install it first.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep ".mp4" as the final suffix so OpenCV/FFmpeg can infer container/codec.
    temp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_output_path.exists():
        temp_output_path.unlink()
    num_frames = ep_end - ep_start
    if num_frames <= 0:
        logging.warning(f"Skip MP4 generation for empty episode: {title}")
        return

    canvas_h, canvas_w = 720, 1280
    frame_area_x, frame_area_y = 30, 50
    frame_area_w, frame_area_h = 620, 540
    chart_x, chart_y = 700, 80
    chart_w, chart_h = 540, 420
    stage_bar_x, stage_bar_y = chart_x, 560
    stage_bar_w, stage_bar_h = chart_w, 28

    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (canvas_w, canvas_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {temp_output_path}")

    cmap = plt.cm.tab10(np.linspace(0, 1, stage_preds.shape[1]))
    stage_colors_bgr = [
        (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        for c in cmap
    ]

    def value_to_chart_y(val: float) -> int:
        clipped = float(np.clip(val, 0.0, 1.0))
        return int(chart_y + chart_h - clipped * chart_h)

    success = False
    try:
        for local_idx in range(num_frames):
            sample = dataset[ep_start + local_idx]
            frame = to_numpy_image(sample[image_key])
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)

            panel = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            panel[:] = (18, 18, 18)

            # Resize and place observation frame in the left panel
            h, w = frame.shape[:2]
            scale = min(frame_area_w / max(w, 1), frame_area_h / max(h, 1))
            draw_w = max(1, int(w * scale))
            draw_h = max(1, int(h * scale))
            resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
            x0 = frame_area_x + (frame_area_w - draw_w) // 2
            y0 = frame_area_y + (frame_area_h - draw_h) // 2
            panel[y0 : y0 + draw_h, x0 : x0 + draw_w] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                panel,
                (frame_area_x - 2, frame_area_y - 2),
                (frame_area_x + frame_area_w + 2, frame_area_y + frame_area_h + 2),
                (80, 80, 80),
                1,
            )

            # Draw chart region and grid
            cv2.rectangle(panel, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (120, 120, 120), 1)
            for g in range(1, 5):
                gy = chart_y + (g * chart_h) // 5
                cv2.line(panel, (chart_x, gy), (chart_x + chart_w, gy), (55, 55, 55), 1)

            # Draw predicted progress line up to current frame
            if local_idx > 0:
                pts = []
                for j in range(local_idx + 1):
                    x = chart_x + int(j * chart_w / max(num_frames - 1, 1))
                    y = value_to_chart_y(float(progress_preds[j]))
                    pts.append([x, y])
                if len(pts) >= 2:
                    cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (255, 170, 60), 2)

            # Optionally overlay GT progress line
            if gt_progress is not None and np.isfinite(gt_progress[local_idx]):
                gt_pts = []
                for j in range(local_idx + 1):
                    if not np.isfinite(gt_progress[j]):
                        continue
                    x = chart_x + int(j * chart_w / max(num_frames - 1, 1))
                    y = value_to_chart_y(float(gt_progress[j]))
                    gt_pts.append([x, y])
                if len(gt_pts) >= 2:
                    cv2.polylines(panel, [np.array(gt_pts, dtype=np.int32)], False, (70, 210, 90), 2)

            # Current time cursor
            cursor_x = chart_x + int(local_idx * chart_w / max(num_frames - 1, 1))
            cv2.line(panel, (cursor_x, chart_y), (cursor_x, chart_y + chart_h), (190, 190, 190), 1)

            current_progress = float(progress_preds[local_idx])
            cv2.putText(
                panel,
                f"frame={local_idx:04d}  progress={current_progress:.3f}",
                (chart_x, chart_y - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

            # Stage probabilities for current frame
            current_stage = np.nan_to_num(stage_preds[local_idx], nan=0.0)
            current_stage = np.clip(current_stage, 0.0, 1.0)
            ssum = float(current_stage.sum())
            if ssum > 0:
                current_stage = current_stage / ssum

            sx = stage_bar_x
            for sid, p in enumerate(current_stage):
                bar_w = int(round(float(p) * stage_bar_w))
                if bar_w <= 0:
                    continue
                color = stage_colors_bgr[sid % len(stage_colors_bgr)]
                cv2.rectangle(
                    panel,
                    (sx, stage_bar_y),
                    (min(stage_bar_x + stage_bar_w, sx + bar_w), stage_bar_y + stage_bar_h),
                    color,
                    -1,
                )
                sx += bar_w

            cv2.rectangle(
                panel,
                (stage_bar_x, stage_bar_y),
                (stage_bar_x + stage_bar_w, stage_bar_y + stage_bar_h),
                (180, 180, 180),
                1,
            )

            top_stage = int(np.argmax(current_stage)) if current_stage.size else 0
            stage_name = stage_labels[top_stage] if top_stage < len(stage_labels) else f"Stage {top_stage + 1}"
            cv2.putText(
                panel,
                f"top_stage: {stage_name}",
                (stage_bar_x, stage_bar_y + stage_bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                title[:80],
                (30, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            writer.write(panel)
        success = True
    finally:
        writer.release()
        if success:
            temp_output_path.replace(output_path)
        elif temp_output_path.exists():
            temp_output_path.unlink()

    print(f"Saved: {output_path}")


def visualize_sarm_predictions(
    dataset: LeRobotDataset,
    reward_model: SARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: Path,
    num_display_frames: int = 5,
    stride: int = 1,
    save_mp4: bool = False,
    mp4_fps: int = 20,
):
    """
    Visualize SARM predictions for multiple episodes.

    Computes predictions for every frame by default. With stride > 1, computes predictions
    every N frames and interpolates (progress + stage probabilities) for visualization.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured
        reward_model: Loaded SARM model
        preprocess: Preprocessor from make_sarm_pre_post_processors
        episode_indices: List of episode indices to visualize
        head_mode: "sparse", "dense", or "both"
        output_dir: Directory to save visualizations
        num_display_frames: Number of frames to display in thumbnail strip (default: 5)
        stride: Compute predictions every N frames, interpolate the rest (default: 1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    dual_mode = reward_model.config.uses_dual_heads
    device = reward_model.device

    # Center frame index for bidirectional sampling
    target_idx = reward_model.config.n_obs_steps // 2

    # Determine which heads to visualize
    schemes_to_viz = []
    if head_mode in ("sparse", "both") or not dual_mode:
        schemes_to_viz.append("sparse")
    if head_mode in ("dense", "both") and dual_mode:
        schemes_to_viz.append("dense")

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        task = dataset[ep_start].get("task", "perform the task")
        num_frames = ep_end - ep_start

        # Select frames for display thumbnails (evenly sampled from begin to end)
        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames
            else list(range(ep_start, ep_end))
        )
        viz_frames = {}

        # Load display frames up-front (stride mode might skip them otherwise).
        for frame_idx in display_indices:
            sample = dataset[frame_idx]
            viz_frames[frame_idx] = to_numpy_image(sample[image_key])

        # Initialize storage for each scheme
        scheme_data = {}
        for scheme in schemes_to_viz:
            num_stages = getattr(reward_model.config, f"num_{scheme}_stages")
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan),
                "viz_stages": np.full((num_frames, num_stages), np.nan),
                "viz_gt_progress": np.full(num_frames, np.nan),
                "viz_gt_stages": np.full(num_frames, np.nan),
                "target_key": f"{scheme}_targets",
                "num_stages": num_stages,
                "temporal_props": getattr(reward_model.config, f"{scheme}_temporal_proportions"),
                "subtask_names": getattr(reward_model.config, f"{scheme}_subtask_names"),
            }

        if stride > 1:
            logging.info(f"Visualization stride={stride}: inferring every {stride} frames and interpolating")

        # Process frames one at a time to avoid memory buildup
        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                image_key: sample[image_key],
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)
                video_features = processed["video_features"].to(device)
                text_features = processed["text_features"].to(device)
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                lengths = processed.get("lengths")

                for scheme in schemes_to_viz:
                    sd = scheme_data[scheme]

                    # Ground truth
                    # In stride visualization mode, ground-truth plots can be misleading
                    # (only sparse points are available), so we skip GT.
                    if stride == 1 and sd["target_key"] in processed:
                        gt_target = processed[sd["target_key"]][0, target_idx].cpu().item()
                        sd["viz_gt_stages"][local_idx] = int(gt_target)
                        sd["viz_gt_progress"][local_idx] = normalize_stage_tau(
                            gt_target,
                            num_stages=sd["num_stages"],
                            temporal_proportions=sd["temporal_props"],
                            subtask_names=sd["subtask_names"],
                        )

                    # Predictions
                    reward, stage_probs = reward_model.calculate_rewards(
                        text_embeddings=text_features,
                        video_embeddings=video_features,
                        state_features=state_features,
                        lengths=lengths,
                        return_all_frames=True,
                        return_stages=True,
                        head_mode=scheme,
                    )

                    # Handle both tensor and numpy outputs
                    if isinstance(reward, torch.Tensor):
                        reward = reward.cpu().numpy()
                        stage_probs = stage_probs.cpu().numpy()

                    if reward.ndim == 2:
                        sd["viz_progress"][local_idx] = reward[0, target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[0, target_idx, :]
                    else:
                        sd["viz_progress"][local_idx] = reward[target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[target_idx, :]

                # Clear GPU memory after each frame
                del processed, video_features, text_features
                if state_features is not None:
                    del state_features

            torch.cuda.empty_cache()

        # Interpolate predictions back to per-frame arrays for smooth visualization.
        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]

                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(
                        valid_idx, sd["viz_progress"][valid_idx], all_local
                    )

                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for s in range(sd["num_stages"]):
                        stage_interp[:, s] = interpolate_progress(
                            valid_idx, sd["viz_stages"][valid_idx, s], all_local
                        )

                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    nz = row_sums.squeeze(-1) > 0
                    stage_interp[nz] = stage_interp[nz] / row_sums[nz]
                    sd["viz_stages"] = stage_interp
                else:
                    # No valid points: keep NaNs/zeros; visualization will be empty.
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        # Generate visualization for each head
        ordered_viz_frames = [viz_frames[idx] for idx in sorted(display_indices)]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            viz_path = output_dir / f"sarm_prediction_ep{episode_idx}_{scheme}.png"
            gt_progress = sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None
            gt_stages = sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None

            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx})",
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=gt_progress,
                gt_stages=gt_stages,
            )
            if save_mp4:
                mp4_path = output_dir / f"sarm_prediction_ep{episode_idx}_{scheme}.mp4"
                save_episode_prediction_mp4(
                    dataset=dataset,
                    image_key=image_key,
                    ep_start=ep_start,
                    ep_end=ep_end,
                    progress_preds=sd["viz_progress"],
                    stage_preds=sd["viz_stages"],
                    stage_labels=stage_labels,
                    output_path=mp4_path,
                    title=f"{task} (Episode {episode_idx}, {scheme})",
                    fps=mp4_fps,
                    gt_progress=gt_progress,
                )

        # Clear memory between episodes
        torch.cuda.empty_cache()

    logging.info(f"Visualizations saved to: {output_dir.absolute()}")


def generate_all_frame_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    """Generate all frame indices, ordered by offset for cache-friendly access.

    Orders frames as: [0, 30, 60...], [1, 31, 61...], ..., [29, 59, 89...]
    This groups frames that share similar temporal windows together.
    """
    num_frames = ep_end - ep_start
    indices = []
    for offset in range(frame_gap):
        for frame_rel in range(offset, num_frames, frame_gap):
            indices.append(ep_start + frame_rel)
    return indices


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate values to fill in gaps (robust to NaNs / edge cases)."""
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def compute_advantage_from_progress(
    progress: np.ndarray,
    horizon: int,
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame advantage deltas and tri-state labels from progress."""
    n = len(progress)
    deltas = np.full(n, np.nan, dtype=np.float32)
    labels = np.full(n, 1, dtype=np.int8)  # 0=regressive, 1=stagnant, 2=progressive
    if n == 0:
        return deltas, labels

    for idx in range(n):
        if not np.isfinite(progress[idx]):
            continue
        future_idx = min(idx + horizon, n - 1)
        if not np.isfinite(progress[future_idx]):
            continue
        delta = float(progress[future_idx] - progress[idx])
        deltas[idx] = delta
        if delta > positive_threshold:
            labels[idx] = 2
        elif delta < negative_threshold:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return deltas, labels


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    num_visualizations: int = 5,
    output_dir: str = "./sarm_viz",
    stride: int = 1,
    save_mp4: bool = False,
    mp4_fps: int = 20,
    advantage_horizon: int = 1,
    advantage_positive_threshold: float = 0.01,
    advantage_negative_threshold: float = -0.01,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        device: Device to use for inference
        num_visualizations: Number of episodes to visualize (0 to skip)
        output_dir: Directory to save visualizations
        stride: Compute progress every N frames, interpolate the rest (default: 1 = every frame)
        advantage_horizon: Frames ahead for advantage delta labels
        advantage_positive_threshold: Delta threshold for progressive labels
        advantage_negative_threshold: Delta threshold for regressive labels
    """
    dataset, reward_model, preprocess = load_sarm_resources(dataset_repo_id, reward_model_path, device)
    if advantage_horizon < 1:
        raise ValueError(f"advantage_horizon must be >= 1, got {advantage_horizon}")

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    logging.info(f"Processing {total_frames} frames across {num_episodes} episodes")

    # Determine which heads to compute
    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    # Storage arrays
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_episode_lengths = []
    all_progress_sparse = [] if compute_sparse else None
    all_progress_dense = [] if compute_dense else None

    if stride > 1:
        logging.info(f"Using stride={stride}: computing every {stride} frames, interpolating the rest")

    # Process all episodes
    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        ep_len = ep_end - ep_start

        # Get task description
        task = dataset[ep_start].get("task", "perform the task")

        # Generate frames to compute (with stride applied)
        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            # Only compute every stride-th frame (relative to episode start)
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            # Always include last frame for better interpolation at episode end
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        center_idx = reward_model.config.n_obs_steps // 2  # Center of bidirectional window

        # Dictionary to collect results
        frame_results = {}

        for query_idx in tqdm(compute_indices, desc=f"  Ep {episode_idx}", leave=False):
            try:
                sample = dataset[query_idx]

                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]

                with torch.no_grad():
                    processed = preprocess(batch)
                    video_features = processed["video_features"].to(device)
                    text_features = processed["text_features"].to(device)
                    state_features = processed.get("state_features")
                    if state_features is not None:
                        state_features = state_features.to(device)
                    lengths = processed.get("lengths")

                    sparse_val = np.nan
                    dense_val = np.nan

                    # Compute sparse prediction for center frame
                    if compute_sparse:
                        sparse_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="sparse",
                        )
                        sparse_val = float(
                            sparse_progress[0, center_idx]
                            if sparse_progress.ndim == 2
                            else sparse_progress[center_idx]
                        )

                    # Compute dense prediction for center frame
                    if compute_dense:
                        dense_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="dense",
                        )
                        dense_val = float(
                            dense_progress[0, center_idx]
                            if dense_progress.ndim == 2
                            else dense_progress[center_idx]
                        )

                    frame_results[query_idx] = (sparse_val, dense_val)

            except Exception as e:
                logging.warning(f"Failed to process frame {query_idx}: {e}")

        # Interpolate to get values for all frames
        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        # All frame indices for this episode
        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            # Interpolate progress values
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            # No interpolation needed
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        # Store results for all frames
        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            all_indices.append(frame_idx)
            all_episode_indices.append(episode_idx)
            all_frame_indices.append(local_idx)
            all_episode_lengths.append(ep_len)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    all_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    all_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    all_progress_dense.append(frame_results[frame_idx][1])
                else:
                    all_progress_dense.append(np.nan)

    # Create output table
    table_data = {
        "index": np.array(all_indices, dtype=np.int64),
        "episode_index": np.array(all_episode_indices, dtype=np.int64),
        "frame_index": np.array(all_frame_indices, dtype=np.int64),
        "episode_length": np.array(all_episode_lengths, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(all_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(all_progress_dense, dtype=np.float32)

    # Sort by index
    df = pa.table(table_data).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)

    # Add advantage deltas + tri-state labels for each available progress head.
    for progress_col, delta_col, label_col in [
        ("progress_sparse", "advantage_delta_sparse", "advantage_label_sparse"),
        ("progress_dense", "advantage_delta_dense", "advantage_label_dense"),
    ]:
        if progress_col not in df.columns:
            continue
        df[delta_col] = np.nan
        df[label_col] = 1  # stagnant
        for episode_idx, group in df.groupby("episode_index", sort=False):
            deltas, labels = compute_advantage_from_progress(
                group[progress_col].to_numpy(dtype=np.float32),
                horizon=advantage_horizon,
                positive_threshold=advantage_positive_threshold,
                negative_threshold=advantage_negative_threshold,
            )
            df.loc[group.index, delta_col] = deltas
            df.loc[group.index, label_col] = labels.astype(np.int8)

    final_table = pa.Table.from_pandas(df, preserve_index=False)

    # Add metadata with reward model path
    metadata = {
        b"reward_model_path": reward_model_path.encode(),
        b"reward_mode": str(getattr(reward_model.config, "reward_mode", "sarm")).encode(),
        b"head_mode": str(head_mode).encode(),
        b"stride": str(stride).encode(),
        b"advantage_horizon": str(advantage_horizon).encode(),
        b"advantage_positive_threshold": str(advantage_positive_threshold).encode(),
        b"advantage_negative_threshold": str(advantage_negative_threshold).encode(),
    }
    final_table = final_table.replace_schema_metadata(metadata)

    # Determine output path
    output_path = Path(dataset.root) / "sarm_progress.parquet" if output_path is None else Path(output_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")

    # Print statistics
    if "progress_sparse" in df.columns:
        valid = df["progress_sparse"].dropna()
        logging.info(
            f"Sparse progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    if "progress_dense" in df.columns:
        valid = df["progress_dense"].dropna()
        logging.info(
            f"Dense progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    # Visualize episodes after processing
    if num_visualizations > 0:
        viz_episodes = list(range(min(num_visualizations, num_episodes)))
        logging.info(f"Generating {len(viz_episodes)} visualizations...")
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
            save_mp4=save_mp4,
            mp4_fps=mp4_fps,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute SARM progress values for RA-BC weighting or visualize SARM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full RA-BC computation with visualizations
    python src/lerobot/policies/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/policies/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 10
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID or local path",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to pretrained SARM model (reads from existing parquet metadata if not provided)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for parquet. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="SARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    # Visualization options
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize SARM predictions (no RA-BC computation)",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5, set to 0 to skip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sarm_viz",
        help="Output directory for visualizations (default: ./sarm_viz)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Upload progress parquet to the HuggingFace dataset repo (requires --dataset-repo-id like org/name, not a local path)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute progress every N frames, interpolate the rest (default: 1 = every frame)",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also export per-episode prediction videos as MP4",
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=20,
        help="FPS for exported prediction MP4 videos (default: 20)",
    )
    parser.add_argument(
        "--advantage-horizon",
        type=int,
        default=1,
        help="Frames ahead used to compute advantage deltas/labels in parquet (default: 1)",
    )
    parser.add_argument(
        "--advantage-positive-threshold",
        type=float,
        default=0.01,
        help="Delta threshold for progressive tri-state labels (default: 0.01)",
    )
    parser.add_argument(
        "--advantage-negative-threshold",
        type=float,
        default=-0.01,
        help="Delta threshold for regressive tri-state labels (default: -0.01)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Try to get reward_model_path from parquet metadata if not provided
    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        # Load dataset to find parquet path
        temp_dataset = LeRobotDataset(args.dataset_repo_id, download_videos=False)
        parquet_path = Path(temp_dataset.root) / "sarm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)"
            )

    # Handle visualize-only mode
    if args.visualize_only:
        dataset, reward_model, preprocess = load_sarm_resources(
            args.dataset_repo_id, reward_model_path, args.device
        )
        logging.info(f"Visualization-only mode: visualizing {args.num_visualizations} episodes")
        viz_episodes = list(range(min(args.num_visualizations, dataset.num_episodes)))
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=args.head_mode,
            output_dir=Path(args.output_dir),
            stride=args.stride,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
        )
        print(f"\nVisualizations saved to: {Path(args.output_dir).absolute()}")
        return

    # Full RABC computation (compute_sarm_progress loads model/dataset itself)
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        save_mp4=args.save_mp4,
        mp4_fps=args.mp4_fps,
        advantage_horizon=args.advantage_horizon,
        advantage_positive_threshold=args.advantage_positive_threshold,
        advantage_negative_threshold=args.advantage_negative_threshold,
    )

    print(f"\nSARM progress values saved to: {output_path}")

    # Upload to Hub if requested (Hub repo id must be org/name, not a filesystem path)
    if args.push_to_hub:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HFValidationError

        api = HfApi()
        hub_path = "sarm_progress.parquet"

        try:
            print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=hub_path,
                repo_id=args.dataset_repo_id,
                repo_type="dataset",
            )
        except HFValidationError as e:
            raise ValueError(
                "Upload failed: --dataset-repo-id must be a HuggingFace dataset id (e.g. org/name), "
                "not a local folder. Omit --push-to-hub when using a local dataset, or pass a valid Hub id."
            ) from e
        print(
            f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
        )

        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print("  rabc_head_mode: sparse  # or dense")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()

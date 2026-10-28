#!/usr/bin/env python3
"""Polygon-level geometry metrics: C-IoU, Vertex F1@3px, and PoLiS.

Metric definitions follow the official SAMPolyBuild and PolyWorld repositories:
- PolyWorld: image IoU and complexity-aware IoU (C-IoU)
- SAMPolyBuild: Boundary F, vertex precision/recall/F1, and PoLiS
- PoLiS: Avbelj et al., "A Metric for Polygon Comparison and Building
  Extraction Evaluation"

Both the ground truth and the predictions are converted to polygons with the
same pipeline: 8-connected components, a minimum component area of 16 px^2,
and Douglas-Peucker simplification with epsilon = 2.0 px (Table I protocol of
the paper). Vertex matching uses a 3 px Euclidean tolerance; PoLiS matches
predicted polygons to ground-truth polygons by bounding-box IoU > 0.5.

Example:
    python evaluate_geometry.py --pred_dir results/WHU/pred_masks --gt_dir /data/WHU/test/masks
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm

GT_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
PRED_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

METRICS = [
    "image_iou",
    "ciou",
    "boundary_f",
    "vertex_precision",
    "vertex_recall",
    "vertex_f1",
    "polis",
]
DISPLAY = {
    "image_iou": "IoU",
    "ciou": "C-IoU",
    "boundary_f": "B-F1@2px",
    "vertex_precision": "Vertex P@3px",
    "vertex_recall": "Vertex R@3px",
    "vertex_f1": "Vertex F1@3px",
    "polis": "PoLiS (px)",
}


def to_binary_numpy(mask, threshold: float = 0.5) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.max() > 1:
        return (mask > 127).astype(np.uint8)
    return (mask > threshold).astype(np.uint8)


def load_image(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


@dataclass
class Components:
    labels: np.ndarray
    areas: np.ndarray
    vertices: list[np.ndarray]

    @property
    def count(self) -> int:
        return len(self.vertices)


def extract_components(mask: np.ndarray, epsilon_px: float, min_area_px: int) -> Components:
    """Extract filtered 8-connected objects and simplified external polygons."""
    binary = to_binary_numpy(mask).astype(np.uint8)
    count, raw_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    areas: list[float] = []
    vertices: list[np.ndarray] = []
    next_label = 1

    for raw_label in range(1, count):
        area = int(stats[raw_label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        component_mask = (raw_labels == raw_label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, epsilon_px, True)
        polygon_vertices = polygon[:, 0, :].astype(np.float32)
        if len(polygon_vertices) < 3:
            continue

        labels[raw_labels == raw_label] = next_label
        areas.append(area)
        vertices.append(polygon_vertices)
        next_label += 1

    return Components(
        labels=labels,
        areas=np.asarray(areas, dtype=np.float64),
        vertices=vertices,
    )


def image_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = to_binary_numpy(pred_mask).astype(bool)
    gt = to_binary_numpy(gt_mask).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def complexity_aware_iou(iou: float, pred_vertices: int, gt_vertices: int) -> float:
    """PolyWorld C-IoU: IoU weighted by the vertex-count similarity."""
    similarity = 1.0 - abs(pred_vertices - gt_vertices) / (pred_vertices + gt_vertices + 1e-9)
    return float(iou * similarity)


def vertex_match_counts(
    pred_vertices: np.ndarray, gt_vertices: np.ndarray, threshold_px: float = 3.0
) -> dict[str, int]:
    """SAMPolyBuild nearest-vertex threshold rule, returned as dataset-level counts."""
    if len(pred_vertices) and len(gt_vertices):
        diff = pred_vertices[:, None, :] - gt_vertices[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        matched_pred = int(np.count_nonzero(np.any(dist_sq < threshold_px**2, axis=1)))
        matched_gt = int(np.count_nonzero(np.any(dist_sq < threshold_px**2, axis=0)))
    else:
        matched_pred = 0
        matched_gt = 0
    return {
        "matched_pred_vertices": matched_pred,
        "matched_gt_vertices": matched_gt,
        "pred_vertices": int(len(pred_vertices)),
        "gt_vertices": int(len(gt_vertices)),
    }


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """DAVIS/SAMPolyBuild one-pixel boundary map."""
    seg = np.asarray(seg).astype(bool)
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    south_east = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    south_east[:-1, :-1] = seg[1:, 1:]
    boundary = seg ^ east | seg ^ south | seg ^ south_east
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def boundary_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """Official SAMPolyBuild/DAVIS Boundary F implementation."""
    pred_boundary = seg2bmap(to_binary_numpy(pred_mask))
    gt_boundary = seg2bmap(to_binary_numpy(gt_mask))
    pred_dilated = binary_dilation(pred_boundary, disk(tolerance_px))
    gt_dilated = binary_dilation(gt_boundary, disk(tolerance_px))

    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count > 0:
        precision, recall = 1.0, 0.0
    elif pred_count > 0 and gt_count == 0:
        precision, recall = 0.0, 1.0
    elif pred_count == 0 and gt_count == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float((pred_boundary * gt_dilated).sum() / pred_count)
        recall = float((gt_boundary * pred_dilated).sum() / gt_count)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _point_to_segments_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from each point to a closed polygon boundary."""
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segment = ends - starts
    length_sq = np.sum(segment * segment, axis=1)

    point_delta = points[:, None, :] - starts[None, :, :]
    projection = np.sum(point_delta * segment[None, :, :], axis=2)
    projection = np.divide(
        projection,
        length_sq[None, :],
        out=np.zeros_like(projection),
        where=length_sq[None, :] > 0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[:, :, None] * segment[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return distances.min(axis=1)


def polis_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """PoLiS (Avbelj et al.): symmetric vertex-to-polygon-boundary distance."""
    a_to_b = _point_to_segments_distance(poly_a, poly_b).mean()
    b_to_a = _point_to_segments_distance(poly_b, poly_a).mean()
    return float(0.5 * (a_to_b + b_to_a))


def _polygon_bboxes(components: Components) -> np.ndarray:
    boxes = []
    for vertices in components.vertices:
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        boxes.append([minimum[0], minimum[1], maximum[0], maximum[1]])
    return np.asarray(boxes, dtype=np.float64)


def _bbox_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    left_top = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    right_bottom = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    pred_area = np.prod(np.maximum(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0), axis=1)
    gt_area = np.prod(np.maximum(gt_boxes[:, 2:] - gt_boxes[:, :2], 0.0), axis=1)
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def polis_image(
    pred: Components, gt: Components, bbox_iou_threshold: float = 0.5
):
    """SAMPolyBuild PoLiS matching: best predicted bbox for each GT, IoU > threshold."""
    if pred.count == 0 or gt.count == 0:
        return np.nan, 0, gt.count
    ious = _bbox_iou(_polygon_bboxes(pred), _polygon_bboxes(gt))
    values = []
    for gt_index, gt_polygon in enumerate(gt.vertices):
        pred_index = int(np.argmax(ious[:, gt_index]))
        if ious[pred_index, gt_index] > bbox_iou_threshold:
            values.append(polis_distance(gt_polygon, pred.vertices[pred_index]))
    if not values:
        return np.nan, 0, gt.count
    return float(np.mean(values)), len(values), gt.count


def _flatten_vertices(components: Components) -> np.ndarray:
    if not components.vertices:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(components.vertices, axis=0).astype(np.float32)


def _init_worker() -> None:
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    cv2.setNumThreads(1)


def _evaluate_pair(task):
    gt_path_str, pred_path_str, config = task
    gt_mask = load_image(gt_path_str)
    pred_mask = load_image(pred_path_str)
    if gt_mask is None:
        raise RuntimeError(f"Cannot read GT: {gt_path_str}")
    if pred_mask is None:
        raise RuntimeError(f"Cannot read prediction: {pred_path_str}")
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    gt_components = extract_components(gt_mask, config["epsilon_px"], config["min_area_px"])
    pred_components = extract_components(pred_mask, config["epsilon_px"], config["min_area_px"])
    gt_vertices = _flatten_vertices(gt_components)
    pred_vertices = _flatten_vertices(pred_components)

    iou = image_iou(pred_mask, gt_mask)
    vertex_counts = vertex_match_counts(pred_vertices, gt_vertices, config["vertex_tolerance_px"])
    polis, polis_matches, polis_gt_objects = polis_image(
        pred_components, gt_components, config["polis_bbox_iou_threshold"]
    )
    return {
        "image_iou": iou,
        "ciou": complexity_aware_iou(iou, len(pred_vertices), len(gt_vertices)),
        "boundary_f": boundary_f_score(pred_mask, gt_mask, config["boundary_tolerance_px"]),
        "polis": polis,
        "polis_matches": float(polis_matches),
        "polis_gt_objects": float(polis_gt_objects),
        **vertex_counts,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)


def evaluate(pred_dir: Path, gt_dir: Path, config: dict, workers: int):
    gt_files = sorted(
        p for p in gt_dir.iterdir() if p.suffix.lower() in GT_EXTS and p.is_file()
    )
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth masks found in {gt_dir}")

    tasks = []
    for gt_path in gt_files:
        pred_path = None
        for ext in PRED_EXTS:
            candidate = pred_dir / f"{gt_path.stem}{ext}"
            if candidate.exists():
                pred_path = candidate
                break
        if pred_path is None:
            raise FileNotFoundError(f"Missing prediction for {gt_path.stem} in {pred_dir}")
        tasks.append((str(gt_path), str(pred_path), config))

    print(f"{len(tasks)} image pairs, workers={workers}")
    if workers > 1 and os.name != "nt":
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("fork"), initializer=_init_worker
        ) as executor:
            results = list(
                tqdm(executor.map(_evaluate_pair, tasks, chunksize=4), total=len(tasks))
            )
    else:
        _init_worker()
        results = [_evaluate_pair(task) for task in tqdm(tasks)]

    summary = {}
    for metric in ["image_iou", "ciou", "boundary_f", "polis"]:
        values = np.asarray([result[metric] for result in results], dtype=float)
        finite = np.isfinite(values)
        summary[metric] = float(np.mean(values[finite])) if finite.any() else float("nan")

    matched_pred = sum(result["matched_pred_vertices"] for result in results)
    matched_gt = sum(result["matched_gt_vertices"] for result in results)
    pred_vertices = sum(result["pred_vertices"] for result in results)
    gt_vertices = sum(result["gt_vertices"] for result in results)
    summary["vertex_precision"] = _safe_ratio(matched_pred, pred_vertices)
    summary["vertex_recall"] = _safe_ratio(matched_gt, gt_vertices)
    summary["vertex_f1"] = (
        0.0
        if summary["vertex_precision"] + summary["vertex_recall"] == 0
        else 2
        * summary["vertex_precision"]
        * summary["vertex_recall"]
        / (summary["vertex_precision"] + summary["vertex_recall"])
    )
    return summary, len(tasks)


def save_report(summary, image_count: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["images", *METRICS])
        writer.writerow(
            [image_count] + [f"{summary[metric]:.8f}" for metric in METRICS]
        )
    print(f"CSV: {output_path}")


def print_summary(summary, image_count: int):
    print("=" * 62)
    print(f"{'Polygon Geometry Metrics':^60}")
    print("=" * 62)
    print(f"  Total successfully processed images: {image_count}")
    print("-" * 62)
    for metric in METRICS:
        value = summary[metric]
        rendered = f"{value:.3f}" if metric == "polis" else f"{value * 100:.2f}%"
        print(f"  {DISPLAY[metric]:<25} | {rendered}")
    print("=" * 62)
    print("Protocol: 8-connected components, min area 16 px^2, Douglas-Peucker")
    print("epsilon 2.0 px; vertex tolerance 3 px; PoLiS bbox-IoU matching > 0.5.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted mask directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground-truth mask directory")
    parser.add_argument("--epsilon_px", type=float, default=2.0)
    parser.add_argument("--min_area_px", type=int, default=16)
    parser.add_argument("--boundary_tolerance_px", type=int, default=2)
    parser.add_argument("--vertex_tolerance_px", type=float, default=3.0)
    parser.add_argument("--polis_bbox_iou_threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    config = {
        "epsilon_px": args.epsilon_px,
        "min_area_px": args.min_area_px,
        "boundary_tolerance_px": args.boundary_tolerance_px,
        "vertex_tolerance_px": args.vertex_tolerance_px,
        "polis_bbox_iou_threshold": args.polis_bbox_iou_threshold,
    }

    summary, image_count = evaluate(Path(args.pred_dir), Path(args.gt_dir), config, args.workers)
    print_summary(summary, image_count)
    if args.output_csv:
        save_report(summary, image_count, Path(args.output_csv))


if __name__ == "__main__":
    main()

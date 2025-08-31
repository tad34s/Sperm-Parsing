import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from __annotated_dataset_fns import Bbox, load_ground_truth_dataset
from mmdet.apis import inference_detector, init_detector
from prepare_images import post_processing


# Add confidence to Bbox or create a new structure
class DetectedBbox:
    def __init__(
        self, x_start: float, x_end: float, y_start: float, y_end: float, confidence: float
    ):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.confidence = confidence


def compute_iou(bbox1: Bbox, bbox2: Bbox) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    inter_x1 = max(bbox1.x_start, bbox2.x_start)
    inter_y1 = max(bbox1.y_start, bbox2.y_start)
    inter_x2 = min(bbox1.x_end, bbox2.x_end)
    inter_y2 = min(bbox1.y_end, bbox2.y_end)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area1 = (bbox1.x_end - bbox1.x_start) * (bbox1.y_end - bbox1.y_start)
    area2 = (bbox2.x_end - bbox2.x_start) * (bbox2.y_end - bbox2.y_start)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area else 0.0


def evaluate_bboxes(
    gt_bboxes: List[Bbox], det_bboxes: List[DetectedBbox], iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate detection accuracy and return metrics including confidence-aware values.
    """
    matches = []
    matched_gts = set()
    matched_dets = set()

    potential_matches = []
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, det_bbox in enumerate(det_bboxes):
            iou = compute_iou(gt_bbox, det_bbox)
            if iou >= iou_threshold:
                potential_matches.append((i, j, iou))

    potential_matches.sort(key=lambda x: x[2], reverse=True)

    for i, j, iou in potential_matches:
        if i not in matched_gts and j not in matched_dets:
            matched_gts.add(i)
            matched_dets.add(j)
            matches.append((i, j, iou))

    TP = len(matches)
    FP = len(det_bboxes) - TP
    FN = len(gt_bboxes) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = sum(iou for _, _, iou in matches) / TP if TP > 0 else 0.0

    return {
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_iou": avg_iou,
    }


def combine_bboxes(bboxes: List[DetectedBbox]) -> List[DetectedBbox]:
    THRESHOLD = 0.5
    n = len(bboxes)
    if n == 0:
        return []

    def should_merge(box1: DetectedBbox, box2: DetectedBbox) -> bool:
        inter_x1 = max(box1.x_start, box2.x_start)
        inter_x2 = min(box1.x_end, box2.x_end)
        inter_y1 = max(box1.y_start, box2.y_start)
        inter_y2 = min(box1.y_end, box2.y_end)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (box1.x_end - box1.x_start) * (box1.y_end - box1.y_start)
        area2 = (box2.x_end - box2.x_start) * (box2.y_end - box2.y_start)
        min_area = min(area1, area2)
        return intersection_area / min_area >= THRESHOLD

    visited = [False] * n
    groups = []
    for i in range(n):
        if not visited[i]:
            group = [i]
            visited[i] = True
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                for j in range(n):
                    if not visited[j] and should_merge(bboxes[idx], bboxes[j]):
                        visited[j] = True
                        queue.append(j)
                        group.append(j)
            groups.append(group)

    merged_bboxes = []
    for group in groups:
        x_start = min(bboxes[i].x_start for i in group)
        x_end = max(bboxes[i].x_end for i in group)
        y_start = min(bboxes[i].y_start for i in group)
        y_end = max(bboxes[i].y_end for i in group)
        confidence = max(bboxes[i].confidence for i in group)
        merged_bboxes.append(DetectedBbox(x_start, x_end, y_start, y_end, confidence))
    return merged_bboxes


def sliding_window(image, window_size, stride):
    win_h, win_w = window_size
    img_h, img_w = image.shape[:2]
    for y in range(0, img_h - win_h + 1, stride):
        for x in range(0, img_w - win_w + 1, stride):
            yield (x, y, image[y : y + win_h, x : x + win_w])


def eval_frame(model, frame_path: Path) -> List[DetectedBbox]:
    frame = cv2.imread(str(frame_path))
    height, width = 200, 200
    stride = 50
    windows = sliding_window(frame, (height, width), stride)
    output_bboxes = []

    for x, y, window in windows:
        ready_window, width_scale, height_scale = post_processing(window)
        ready_window = cv2.cvtColor(ready_window, cv2.COLOR_GRAY2BGR)
        result = inference_detector(model, ready_window)
        bbox_result = result[0] if isinstance(result, tuple) else result
        bboxes = np.vstack(bbox_result) if len(bbox_result) > 0 else np.array([])
        for box in bboxes:
            x1, y1, x2, y2, conf = box
            output_bboxes.append(
                DetectedBbox(
                    x + x1 / width_scale,
                    x + x2 / width_scale,
                    y + y1 / height_scale,
                    y + y2 / height_scale,
                    conf,
                )
            )
    return output_bboxes


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])


def calculate_average_precision(all_detections, all_ground_truths, iou_threshold=0.5):
    detections = []
    for img_id, dets in all_detections.items():
        for det in dets:
            detections.append((img_id, det))
    detections.sort(key=lambda x: x[1].confidence, reverse=True)

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    matched_gt = {img_id: set() for img_id in all_ground_truths.keys()}

    for i, (img_id, det) in enumerate(detections):
        if img_id not in all_ground_truths:
            fp[i] = 1
            continue

        gt_bboxes = all_ground_truths[img_id]
        best_iou = 0
        best_gt_idx = -1
        for idx, gt_bbox in enumerate(gt_bboxes):
            iou = compute_iou(gt_bbox, det)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_id]:
            tp[i] = 1
            matched_gt[img_id].add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / sum(len(gt) for gt in all_ground_truths.values())
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    return compute_ap(recalls, precisions)


def main():
    parser = argparse.ArgumentParser(description="Run object detection inference")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(args.config, args.checkpoint, device=device)

    frame_location = Path("data/eval/frames")
    outputs_location = Path("data/eval/bboxed_frames")
    outputs_location.mkdir(exist_ok=True, parents=True)

    all_detections = {}
    all_ground_truths = {}

    for frame in frame_location.glob("*.jpg"):
        print(f"Processing frame {frame.name}...")
        detected_bboxes = eval_frame(model, frame)
        combined_bboxes = combine_bboxes(detected_bboxes)
        all_detections[frame.stem] = combined_bboxes

        frame_name = frame.stem
        ground_truth_xml = frame_location / f"{frame_name}.xml"
        if ground_truth_xml.exists():
            ground_truth = load_ground_truth_dataset(ground_truth_xml)
            all_ground_truths[frame.stem] = ground_truth
            metrics = evaluate_bboxes(ground_truth, combined_bboxes)
            print(f"Metrics for {frame_name}: {metrics}")

        bboxed_frame = vizualize_bboxes(combined_bboxes, frame)
        cv2.imwrite(str(outputs_location / frame.name), bboxed_frame)

    if all_ground_truths:
        ap = calculate_average_precision(all_detections, all_ground_truths)
        print(f"Average Precision: {ap:.4f}")


if __name__ == "__main__":
    main()

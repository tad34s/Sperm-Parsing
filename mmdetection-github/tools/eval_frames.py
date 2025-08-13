import argparse
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from mmdet.apis import inference_detector, init_detector
from prepare_images import post_processing


@dataclass
class Bbox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int


def compute_iou(bbox1: Bbox, bbox2: Bbox) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    # Calculate intersection coordinates
    inter_x1 = max(bbox1.x_start, bbox2.x_start)
    inter_y1 = max(bbox1.y_start, bbox2.y_start)
    inter_x2 = min(bbox1.x_end, bbox2.x_end)
    inter_y2 = min(bbox1.y_end, bbox2.y_end)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate individual areas
    area1 = (bbox1.x_end - bbox1.x_start) * (bbox1.y_end - bbox1.y_start)
    area2 = (bbox2.x_end - bbox2.x_start) * (bbox2.y_end - bbox2.y_start)

    # Calculate union area
    union_area = area1 + area2 - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def evaluate_bboxes(
    gt_bboxes: List[Bbox], det_bboxes: List[Bbox], iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate detection accuracy using multiple metrics.

    Args:
        gt_bboxes: List of ground truth bounding boxes
        det_bboxes: List of detected bounding boxes
        iou_threshold: Minimum IoU to consider a match (default=0.5)

    Returns:
        Dictionary containing:
        - 'true_positives': Number of correctly matched detections
        - 'false_positives': Number of incorrect detections
        - 'false_negatives': Number of missed ground truths
        - 'precision': Precision score (TP / (TP + FP))
        - 'recall': Recall score (TP / (TP + FN))
        - 'f1_score': F1 score (harmonic mean of precision and recall)
        - 'average_iou': Average IoU of matched pairs
    """
    # Initialize counters and lists
    matches: List[Tuple[int, int, float]] = []
    matched_gts = set()
    matched_dets = set()

    # Find all potential matches above IoU threshold
    potential_matches = []
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, det_bbox in enumerate(det_bboxes):
            iou = compute_iou(gt_bbox, det_bbox)
            if iou >= iou_threshold:
                potential_matches.append((i, j, iou))

    # Sort matches by IoU in descending order
    potential_matches.sort(key=lambda x: x[2], reverse=True)

    # Perform greedy matching (highest IoU first)
    for i, j, iou in potential_matches:
        if i not in matched_gts and j not in matched_dets:
            matched_gts.add(i)
            matched_dets.add(j)
            matches.append((i, j, iou))

    # Calculate metrics
    TP = len(matches)
    FP = len(det_bboxes) - TP
    FN = len(gt_bboxes) - TP

    # Handle edge cases for precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0

    # Calculate F1 score
    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate average IoU for matched pairs
    avg_iou = 0.0
    if TP > 0:
        avg_iou = sum(iou for _, _, iou in matches) / TP

    return {
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_iou": avg_iou,
    }


def load_ground_truth_dataset(xml_path: Path) -> List[Bbox]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bboxes.append(Bbox(xmin, xmax, ymin, ymax))

    return bboxes


def combine_bboxes(bboxes: List[Bbox]) -> List[Bbox]:
    THRESHOLD = 0.5

    n = len(bboxes)
    if n == 0:
        return []

    def should_merge(box1: Bbox, box2: Bbox) -> bool:
        x_start_i = max(box1.x_start, box2.x_start)
        x_end_i = min(box1.x_end, box2.x_end)
        y_start_i = max(box1.y_start, box2.y_start)
        y_end_i = min(box1.y_end, box2.y_end)

        if x_end_i <= x_start_i or y_end_i <= y_start_i:
            return False

        intersection_area = (x_end_i - x_start_i) * (y_end_i - y_start_i)

        area1 = (box1.x_end - box1.x_start) * (box1.y_end - box1.y_start)
        area2 = (box2.x_end - box2.x_start) * (box2.y_end - box2.y_start)

        if area1 <= 0 or area2 <= 0:
            return False

        min_area = min(area1, area2)
        overlap_ratio = intersection_area / min_area

        return overlap_ratio >= THRESHOLD

    visited = [False] * n
    groups = []

    for i in range(n):
        if not visited[i]:
            group = [bboxes[i]]
            visited[i] = True
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                for j in range(n):
                    if not visited[j]:
                        if should_merge(bboxes[idx], bboxes[j]):
                            visited[j] = True
                            queue.append(j)
                            group.append(bboxes[j])
            groups.append(group)

    merged_bboxes = []
    for group in groups:
        x_start = min(bbox.x_start for bbox in group)
        x_end = max(bbox.x_end for bbox in group)
        y_start = min(bbox.y_start for bbox in group)
        y_end = max(bbox.y_end for bbox in group)
        merged_bboxes.append(Bbox(x_start, x_end, y_start, y_end))

    return merged_bboxes


def sliding_window(image, window_size, stride):
    """
    Generator that yields image windows and their coordinates as it slides over the input image.

    Args:
        image: Input image (numpy array)
        window_size: Tuple (height, width) of the window dimensions
        stride: Integer stride (step size) for both horizontal and vertical directions
    Yields:
        (x, y, window): Top-left coordinates (x, y) and the image window
    """
    win_h, win_w = window_size
    img_h, img_w = image.shape[:2]

    for y in range(0, img_h - win_h + 1, stride):
        for x in range(0, img_w - win_w + 1, stride):
            yield (x, y, image[y : y + win_h, x : x + win_w])


def eval_frame(model, frame_path: Path) -> List[Bbox]:
    frame = cv2.imread(str(frame_path))
    height = 200
    width = 200
    stride = 50
    windows = sliding_window(image=frame, window_size=(height, width), stride=stride)

    output_bboxes = []

    for x, y, window in windows:
        ready_window, width_scale, height_scale = post_processing(window)
        ready_window = cv2.cvtColor(ready_window, cv2.COLOR_GRAY2BGR)
        result = inference_detector(model, ready_window)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        for box in bboxes:
            # Convert coordinates to integers
            x1, y1, x2, y2, _ = box

            output_bboxes.append(
                Bbox(
                    x + int(x1) / width_scale,
                    x + int(x2) / width_scale,
                    y + int(y1) / height_scale,
                    y + int(y2) / height_scale,
                )
            )

    return output_bboxes


def vizualize_bboxes(bboxes: List[Bbox], frame: Path) -> MatLike:
    img = cv2.imread(str(frame))
    for box in bboxes:
        cv2.rectangle(
            img,
            (int(box.x_start), int(box.y_start)),
            (int(box.x_end), int(box.y_end)),
            color=(255, 0, 0),  # Red in RGB
            thickness=2,
        )
    return img


if __name__ == "__main__":
    frame_location = Path("data/eval/frames")
    outputs_location = Path("data/eval/bboxed_frames")
    outputs_location.mkdir(exist_ok=True, parents=True)

    parser = argparse.ArgumentParser(description="Run object detection inference")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")

    args = parser.parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(config_file, checkpoint_file, device=device)

    frame_path = next(frame_location.glob("*.jpg"))
    frame = cv2.imread(str(frame_path))
    print("Image shape: ", frame.shape)

    for frame in frame_location.glob("*.jpg"):
        print(f"Processing frame {frame.name}...")
        output_frame = outputs_location / frame.name
        bboxes = eval_frame(model, frame)
        bboxes = combine_bboxes(bboxes)
        frame_name = frame.name[:-4]  # remove .jpg
        ground_truth_xml = frame_location / f"{frame_name}.xml"
        print(f"Detected {len(bboxes)} bboxes...")
        if ground_truth_xml.exists():
            ground_truth = load_ground_truth_dataset(ground_truth_xml)
            values = evaluate_bboxes(ground_truth, bboxes)
            print(f"Metrics for {frame_name}")
            print(values)
            print("------------------------")
        else:
            print(f"{str(ground_truth_xml)} does not exists.")

        bboxed_frame = vizualize_bboxes(bboxes, frame)
        cv2.imwrite(str(output_frame), bboxed_frame)

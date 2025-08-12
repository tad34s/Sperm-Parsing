import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from mmdet.apis import inference_detector, init_detector
from prepare_images import post_processing
from tqdm import tqdm


@dataclass
class Bbox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int


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
    height = 150
    width = 150
    stride = 50
    windows = sliding_window(image=frame, window_size=(height, width), stride=stride)

    output_bboxes = []

    for x, y, window in windows:
        ready_window, width_scale, height_scale = post_processing(window)
        result = inference_detector(model, window)
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
                    x + int(x1) * width_scale,
                    x + int(x2) * width_scale,
                    y + int(y1) * height_scale,
                    y + int(y2) * height_scale,
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

    for frame in tqdm(frame_location.glob("*.jpg")):
        output_frame = outputs_location / frame.name
        bboxes = eval_frame(model, frame)
        bboxes = combine_bboxes(bboxes)
        bboxed_frame = vizualize_bboxes(bboxes, frame)
        cv2.imwrite(str(output_frame), bboxed_frame)

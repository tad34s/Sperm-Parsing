import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

from .prepare_images import post_processing


@dataclass
class Bbox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int


def combine_bboxes(bboxes: list[Bbox]) -> list[Bbox]:
    output_bboxes = []
    return bboxes


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


def eval_frame(model, frame_path: Path) -> list[Bbox]:
    height = 350
    width = 350
    stride = 100
    frame = cv2.imread(str(frame_path))
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


if __name__ == "__main__":
    frame_location = Path("dataset/eval/frames")
    outputs_location = Path("dataset/eval/bboxed_frames")
    outputs_location.mkdir(exist_ok=True, parents=True)

    parser = argparse.ArgumentParser(description="Run object detection inference")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--image", required=True, help="Path to input image")

    args = parser.parse_args()
    config_file = (args.config,)
    checkpoint_file = (args.checkpoint,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_detector(config_file, checkpoint_file, device=device)

    for frame in frame_location.glob("*.jpg"):
        output_frame = outputs_location / frame.name
        bboxes = eval_frame(model, frame)
        bboxes = combine_bboxes(bboxes)
        bboxed_frame = vizualize_bboxes(bboxes, frame)
        cv2.imwrite(str(output_frame), bboxed_frame)

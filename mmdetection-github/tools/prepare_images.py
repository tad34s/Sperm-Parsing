import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def adjust_background_brightness(image, target_bg, percentile=90):
    """
    Adjusts bright background regions to a target brightness level.

    Args:
        image: Input image (BGR color or grayscale)
        target_bg: Target brightness value (0-255) for background
        percentile: Percentile value to identify bright regions

    Returns:
        Adjusted image
    """
    # Convert to HSV if color, or use grayscale directly
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
    else:
        v_channel = image.astype(np.float32)

    # Calculate brightness thresholds
    p_low = np.percentile(v_channel, percentile)  # Brightness threshold
    p_high = np.percentile(v_channel, 100)  # Maximum brightness

    # Avoid adjustment if no bright pixels
    if p_high <= p_low:
        return image.copy()

    # Create mask for bright regions
    mask = v_channel > p_low

    # Linearly scale bright regions: [p_low, p_high] -> [p_low, target_bg]
    # scale = (target_bg - p_low) / (p_high - p_low)
    adjusted_v = np.where(mask, target_bg, v_channel)
    adjusted_v = np.clip(adjusted_v, 0, 255).astype(np.uint8)

    # Merge back to original image
    if image.ndim == 3:
        hsv[:, :, 2] = adjusted_v
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        result = adjusted_v

    return result


def adjust_dark_spots(image, threshold=130, factor=0.70):
    """
    Darkens dark spots (≤ threshold) in a grayscale image with smooth falloff.

    Args:
        image: Input grayscale image (numpy array)
        threshold: Pixels ≤ this value will be adjusted (default=80)
        factor: Darkening strength (0.0-1.0), lower = darker (default=0.92)

    Returns:
        Adjusted grayscale image
    """
    # Create lookup table
    i = np.arange(256)
    table = i.copy().astype(np.float32)  # Start with identity mapping

    # Calculate adjustment for pixels ≤ threshold
    mask = i <= threshold
    table[mask] = i[mask] * factor

    # Clip, round, and convert to uint8
    table = np.clip(table, 0, 255)
    table = np.round(table).astype(np.uint8)

    return cv2.LUT(image, table)


def block_based_elastic_transform(image, block_size=60, margin=15, alpha=6, sigma=4):
    """Apply elastic transformation to image blocks with blending"""
    H, W = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros_like(image, dtype=np.float32)

    # Create Hanning window for smooth blending
    hann = np.outer(np.hanning(block_size), np.hanning(block_size))

    # Pad image to handle border blocks
    padded = cv2.copyMakeBorder(
        image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
    )

    # Process in sliding window fashion
    for y in range(0, H, block_size // 2):  # 50% overlap
        for x in range(0, W, block_size // 2):
            # Extract block with margin
            y_start = y
            x_start = x
            block = padded[
                y_start : y_start + block_size + 2 * margin,
                x_start : x_start + block_size + 2 * margin,
            ]

            if block.size == 0:
                continue

            # Apply elastic transformation to block
            rand_state = np.random.RandomState()
            dx = rand_state.rand(*block.shape) * 2 - 1
            dy = rand_state.rand(*block.shape) * 2 - 1
            dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

            X, Y = np.meshgrid(np.arange(block.shape[1]), np.arange(block.shape[0]))
            map_x = np.clip(X + dx, 0, block.shape[1] - 1).astype(np.float32)
            map_y = np.clip(Y + dy, 0, block.shape[0] - 1).astype(np.float32)

            distorted_block = cv2.remap(
                block,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            # Extract inner region (without margin)
            inner_block = distorted_block[
                margin : margin + block_size, margin : margin + block_size
            ]

            # Calculate valid region in output
            y_end = min(y + block_size, H)
            x_end = min(x + block_size, W)
            valid_height = y_end - y
            valid_width = x_end - x

            # Apply weighted blending
            if valid_height > 0 and valid_width > 0:
                valid_mask = hann[:valid_height, :valid_width]
                output[y:y_end, x:x_end] += (
                    inner_block[:valid_height, :valid_width] * valid_mask
                )
                weights[y:y_end, x:x_end] += valid_mask

    # Normalize blended image
    output = np.divide(output, weights, out=np.zeros_like(output), where=weights > 0)
    return output.clip(0, 255).astype(np.uint8)


def post_processing(image):
    # resize
    new_height = 350
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = adjust_background_brightness(image, target_bg=255, percentile=5)
    # blur
    image = cv2.GaussianBlur(image, (15, 15), 2.0)
    kernel = np.ones((3, 3))
    image = cv2.dilate(image, kernel)

    image = block_based_elastic_transform(
        image,
        block_size=60,  # Optimal for sperm cell sizes
        margin=20,  # Context margin for natural distortions
        alpha=40,  # Higher distortion for irregular shapes
        sigma=4,  # Smooth deformations
    )

    # Add noise (lower intensity for better realism)
    noise = np.random.normal(0, 8, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, width_scale, height_scale


def scale_annotation(annotation, width_scale, height_scale):
    """Scales all coordinates in an annotation while preserving structure"""
    scaled = annotation.copy()

    # Scale segmentation components
    for seg_type in ["vacuole", "acrosome", "nucleus", "midpiece", "tail"]:
        key = f"segmentation_{seg_type}"
        original_polygons = annotation[key]

        # Handle both empty and non-empty cases
        if not original_polygons:
            scaled[key] = []
            continue

        # Scale each polygon while preserving the flat list structure
        scaled_polygons = []
        for poly in original_polygons:
            if not poly:  # Skip empty polygons
                continue

            # Scale each coordinate pair
            scaled_poly = []
            for i in range(0, len(poly)):
                # For flat list structure: [x1, y1, x2, y2, ...]
                if i % 2 == 0:  # x-coordinate
                    scaled_poly.append(poly[i] * width_scale)
                else:  # y-coordinate
                    scaled_poly.append(poly[i] * height_scale)
            scaled_polygons.append(scaled_poly)

        scaled[key] = scaled_polygons

    # Scale bounding box
    bbox = annotation["bbox"]
    scaled_bbox = [
        bbox[0] * width_scale,
        bbox[1] * height_scale,
        bbox[2] * width_scale,
        bbox[3] * height_scale,
    ]
    scaled["bbox"] = scaled_bbox

    # Scale area
    scaled["area"] = annotation["area"] * width_scale * height_scale

    return scaled


def construct_bbox(
    segmentation_vacuole: list,
    segmentation_acrosome: list,
    segmentation_nucleus: list,
    segmentation_midpiece: list,
) -> Tuple[List[float], float]:
    max_x = float("-inf")
    max_y = float("-inf")
    min_x = float("inf")
    min_y = float("inf")

    area = 0.0

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    for piece_segm in [
        segmentation_vacuole,
        segmentation_acrosome,
        segmentation_nucleus,
        segmentation_midpiece,
    ]:
        for poly in piece_segm:
            if not poly:  # Skip empty polygons
                continue

            # Extract coordinates from flat list format [x1,y1,x2,y2,...]
            x_coords = []
            y_coords = []
            for i in range(0, len(poly), 2):
                if i + 1 < len(poly):
                    x_coords.append(poly[i])
                    y_coords.append(poly[i + 1])

            if not x_coords or not y_coords:
                continue

            max_x = max(max(x_coords), max_x)
            max_y = max(max(y_coords), max_y)
            min_x = min(min(x_coords), min_x)
            min_y = min(min(y_coords), min_y)
            area += PolyArea(x_coords, y_coords)

    return [
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    ], area


def delete_tail(annotation: dict) -> dict:
    segmentation_vacuole = annotation["segmentation_vacuole"]
    segmentation_acrosome = annotation["segmentation_acrosome"]
    segmentation_nucleus = annotation["segmentation_nucleus"]
    segmentation_midpiece = annotation["segmentation_midpiece"]
    area = annotation["area"]
    bbox = annotation["bbox"]

    output = annotation.copy()

    bbox, area = construct_bbox(
        segmentation_vacuole,
        segmentation_acrosome,
        segmentation_nucleus,
        segmentation_midpiece,
    )
    output["segmentation_tail"] = []
    output["area"] = area
    output["bbox"] = bbox

    return output


def prepare_annotation(
    annotations_file: dict, dest: Path, scaling_factors: dict
) -> None:
    altered_annotations = []
    for annotation in annotations_file["annotations"]:
        no_tail = delete_tail(annotation)

        # Apply scaling if factors exist
        img_id = no_tail["image_id"]
        if img_id in scaling_factors:
            width_scale, height_scale = scaling_factors[img_id]
            scaled = scale_annotation(no_tail, width_scale, height_scale)
            altered_annotations.append(scaled)
        else:
            altered_annotations.append(no_tail)

    annotations_file["annotations"] = altered_annotations

    # Update image dimensions in metadata
    for img in annotations_file["images"]:
        img_id = img["id"]
        if img_id in scaling_factors:
            width_scale, height_scale = scaling_factors[img_id]
            img["width"] = int(img["width"] * width_scale)
            img["height"] = int(img["height"] * height_scale)

    with dest.open("w") as f:
        json.dump(annotations_file, fp=f)


def prepare_image(src: Path, dest: Path) -> Tuple[float, float]:
    image = cv2.imread(str(src))
    final_image, width_scale, height_scale = post_processing(image)
    cv2.imwrite(str(dest), final_image)

    # Extract image ID from filename
    return (width_scale, height_scale)


def prepare_dataset(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(exist_ok=True, parents=True)
    scaling_factors = {}  # {image_id: (width_scale, height_scale)}
    with (src_dir / "annotations.json").open("r") as f:
        annotations_file = json.load(f)

    # Process images and collect scaling factors
    dir_images = src_dir / "JPEGImages"
    for image_data in annotations_file["images"]:
        relative = Path(image_data["file_name"])
        dest_image = dest_dir / relative
        dest_image.parent.mkdir(exist_ok=True, parents=True)

        factors = prepare_image(src_dir / relative, dest_image)
        scaling_factors[image_data["id"]] = factors

    prepare_annotation(
        annotations_file.copy(),
        dest_dir / "annotations.json",
        scaling_factors,
    )


if __name__ == "__main__":
    dest_dir = Path.cwd() / "mmdetection-github" / "data" / "part100x"
    src_dir = Path.cwd() / "mmdetection-github" / "data" / "spermparsing" / "Training"
    prepare_dataset(src_dir, dest_dir)

    src_image = Path.cwd() / "mmdetection-github" / "data" / "eval" / "image3.jpg"
    dest_image = (
        Path.cwd() / "mmdetection-github" / "data" / "eval" / "edited_image3.jpg"
    )
    prepare_image(src_image, dest_image)  # Scaling factors discarded for eval

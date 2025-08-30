from pathlib import Path

import cv2
import numpy as np


def adjust_background_brightness(image, target_bg=200, percentile=90):
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


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    # Generate random displacement fields
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1

    # Apply Gaussian blur to displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Apply displacements
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Remap image using displacement fields
    distorted_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # Use white for borders
    )
    return distorted_image


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
                output[y:y_end, x:x_end] += inner_block[:valid_height, :valid_width] * valid_mask
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

    # image = block_based_elastic_transform(
    #     image,
    #     block_size=60,  # Optimal for sperm cell sizes
    #     margin=20,  # Context margin for natural distortions
    #     alpha=40,  # Higher distortion for irregular shapes
    #     sigma=4,  # Smooth deformations
    # )

    # Add noise (lower intensity for better realism)
    # noise = np.random.normal(0, 8, image.shape).astype(np.float32)
    # image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, width_scale, height_scale


def update_one_image(src: Path, dest: Path) -> None:
    image = cv2.imread(str(src))
    final_image, _, _ = post_processing(image)
    cv2.imwrite(str(dest), final_image)


def update_directory(dir_src: Path, dir_dest: Path):
    dir_dest.mkdir(parents=True, exist_ok=True)
    dir_images = dir_src / "JPEGImages"
    for file in dir_images.glob("*.jpg"):
        relative_path_file = file.relative_to(dir_src)
        dest_path_file = dir_dest / relative_path_file
        dest_path_file.parent.mkdir(parents=True, exist_ok=True)
        update_one_image(file, dest_path_file)


if __name__ == "__main__":
    data_location = Path.cwd() / "sperm_edit"
    src_image = data_location / "example.jpg"
    dest_dir = data_location / "result2.jpg"
    update_one_image(src_image, dest_dir)

    src_image = data_location / "image.jpg"
    dest_dir = data_location / "result.jpg"
    update_one_image(src_image, dest_dir)

    src_image = data_location / "sperms2.jpg"
    dest_dir = data_location / "result3.jpg"
    update_one_image(src_image, dest_dir)

    src_image = data_location / "sperms3.jpg"
    dest_dir = data_location / "result4.jpg"
    update_one_image(src_image, dest_dir)

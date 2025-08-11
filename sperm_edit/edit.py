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


def post_processing(image):
    # resize
    new_height = 350
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = adjust_background_brightness(image, target_bg=255, percentile=5)
    # blur
    image = cv2.GaussianBlur(image, (15, 15), 2.0)

    kernel = np.ones((3, 3))
    image = cv2.dilate(image, kernel)
    image = adjust_dark_spots(image)

    return image


def update_one_image(src: Path, dest: Path) -> None:
    image = cv2.imread(str(src))
    final_image = post_processing(image)
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
    src_image = Path.cwd() / "example.jpg"
    dest_dir = Path.cwd() / "result2.jpg"
    update_one_image(src_image, dest_dir)

    src_image = Path.cwd() / "image.jpg"
    dest_dir = Path.cwd() / "result.jpg"
    update_one_image(src_image, dest_dir)

    src_image = Path.cwd() / "sperms2.jpg"
    dest_dir = Path.cwd() / "result3.jpg"
    update_one_image(src_image, dest_dir)

    src_image = Path.cwd() / "sperms3.jpg"
    dest_dir = Path.cwd() / "result4.jpg"
    update_one_image(src_image, dest_dir)

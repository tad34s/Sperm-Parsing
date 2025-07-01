import shutil
from pathlib import Path

import cv2
import numpy as np


def post_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    # sharpen
    image = cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (1, 1), 5), -0.5, 0)

    new_height = 140

    # resize
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # add noise
    mean = 0
    stddev = 5
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

    image = cv2.add(image, noise)

    # blur
    image = cv2.GaussianBlur(image, (5, 5), 5)

    # resize back to normal
    image = cv2.resize(
        image,
        (original_width, original_height),
        interpolation=cv2.INTER_AREA,
    )
    return image


def update_one_image(src: Path, dest: Path) -> None:
    image = cv2.imread(str(src))
    final_image = post_processing(image)
    cv2.imwrite(str(dest), final_image)


def update_directory(dir_src: Path, dir_dest: Path) -> None:
    shutil.copytree(dir_src, dir_dest)
    dir_images = dir_dest / "JPEGImages"
    for file in dir_images.glob("*.jpg"):
        update_one_image(file, file)


if __name__ == "__main__":
    dest_dir = Path.cwd() / "data" / "part100x"
    src_dir = Path.cwd() / "data" / "spermparsing" / "Training"
    update_directory(src_dir, dest_dir)

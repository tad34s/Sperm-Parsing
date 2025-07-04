import json
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
    # mean = 0
    # stddev = 5
    # noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    # image = cv2.add(image, noise)

    # blur
    image = cv2.GaussianBlur(image, (5, 5), 5)
    # resize back to normal
    image = cv2.resize(
        image,
        (original_width, original_height),
        interpolation=cv2.INTER_AREA,
    )

    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return image


def construct_bbox(
    segmentation_vacuole: list,
    segmentation_acrosome: list,
    segmentation_nucleus: list,
    segmentation_midpiece: list,
) -> tuple[list[float], float]:
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
            if piece_segm is None:
                continue
            x_cords = poly[::2]
            y_cords = poly[1::2]

            max_x = max(x_cords + [max_x])
            max_y = max(y_cords + [max_y])
            min_x = min(x_cords + [min_x])
            min_y = min(y_cords + [min_y])
            area += PolyArea(x_cords, y_cords)

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


def prepare_annotation(src: Path, dest: Path) -> None:
    with src.open("r") as f:
        annotations_file = json.load(f)
    annotations = annotations_file["annotations"]

    altered_annotations = []
    for annotation in annotations:
        altered_annotations.append(delete_tail(annotation))

    annotations_file["annotations"] = altered_annotations

    with dest.open("w") as f:
        json.dump(annotations_file, fp=f)


def prepare_image(src: Path, dest: Path) -> None:
    image = cv2.imread(str(src))
    final_image = post_processing(image)
    cv2.imwrite(str(dest), final_image)


def prepare_dataset(dir_src: Path, dir_dest: Path) -> None:
    dir_images = dir_src / "JPEGImages"
    for file in dir_images.glob("*.jpg"):
        relative = file.relative_to(dir_src)
        dest_image = dir_dest / relative
        dest_image.parent.mkdir(exist_ok=True, parents=True)
        prepare_image(file, dest_image)

    prepare_annotation(dir_src / "annotations.json", dir_dest / "annotations.json")


if __name__ == "__main__":
    dest_dir = Path.cwd() / "data" / "part100x"
    src_dir = Path.cwd() / "data" / "spermparsing" / "Training"
    prepare_dataset(src_dir, dest_dir)

    src_image = Path.cwd() / "data" / "eval" / "image3.jpg"
    dest_image = Path.cwd() / "data" / "eval" / "edited_image3.jpg"
    prepare_image(src_image, dest_image)

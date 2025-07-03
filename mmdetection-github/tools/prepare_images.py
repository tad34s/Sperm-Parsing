import json
import shutil
from pathlib import Path

import cv2


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


def prepare_annotation(src: Path, dest: Path) -> None:
    with src.open("r") as f:
        annotation = json.load(f)
    shapes = annotation["shapes"]

    filtered_shapes = []
    for shape in shapes:
        if shape["label"] == "Tail":
            continue
        filtered_shapes.append(shape)

    annotation["shapes"] = filtered_shapes

    with dest.open("w") as f:
        json.dump(annotation, fp=f)


def prepare_image(src: Path, dest: Path) -> None:
    image = cv2.imread(str(src))
    final_image = post_processing(image)
    cv2.imwrite(str(dest), final_image)


def prepare_dataset(dir_src: Path, dir_dest: Path) -> None:
    shutil.copytree(dir_src, dir_dest)
    dir_images = dir_dest / "JPEGImages"
    for file in dir_images.glob("*.jpg"):
        prepare_image(file, file)

    for file in dir_images.glob("*.json"):
        prepare_annotation(file, file)


if __name__ == "__main__":
    dest_dir = Path.cwd() / "data" / "part100x"
    src_dir = Path.cwd() / "data" / "spermparsing" / "Training"
    prepare_dataset(src_dir, dest_dir)

    src_image = Path.cwd() / "data" / "eval" / "image3.jpg"
    dest_image = Path.cwd() / "data" / "eval" / "edited_image3.jpg"
    prepare_image(src_image, dest_image)

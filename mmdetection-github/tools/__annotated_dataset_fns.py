import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Bbox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int


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

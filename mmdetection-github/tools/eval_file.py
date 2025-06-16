from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

# Paths to your configuration file and pretrained weights
CONFIG_FILE = "configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py"  # Replace with the actual path to your config file
CHECKPOINT_FILE = "epoch_35.pth"  # Replace with the actual path to your .pth file
IMAGE_PATH = "data/eval/image.jpg"  # Path to the image you want to test
OUTPUT_PATH = "data/eval/output.jpg"  # Path to save the visualization result (optional)


def visualize_segments(image, bboxes, segms, palette):
    """
    Visualize all segments for each bounding box.

    Args:
        image (str): Path to the input image.
        bboxes (np.ndarray): Bounding boxes.
        segms (list): List of masks for each segment.
        palette (list): List of colors for visualization.

    Returns:
        np.ndarray: Image with visualized segments.
    """
    img = mmcv.imread(image).astype(np.uint8)
    for i, bbox in enumerate(bboxes):
        if i < len(segms):  # Ensure index is within bounds
            for j, mask in enumerate(segms[i]):
                color_mask = palette[j % len(palette)]
                mask = mask.astype(bool)
                img[mask] = img[mask] * 0.5 + np.array(color_mask, dtype=np.uint8) * 0.5
    return img


def test_image(
    config_file, checkpoint_file, image_path, output_path=None, device="cuda:0"
):
    """
    Run inference on a new image using a pretrained model and visualize the results.

    Args:
        config_file (str): Path to the model configuration file.
        checkpoint_file (str): Path to the pretrained weights (.pth file).
        image_path (str): Path to the image for testing.
        output_path (str, optional): Path to save the visualized result. Defaults to None.
        device (str): Device to run the model on ('cuda:0' for GPU or 'cpu'). Defaults to 'cuda:0'.
    """
    # Initialize the model
    model = init_detector(config_file, checkpoint_file, device=device)

    # Perform inference on the image
    result = inference_detector(model, image_path)

    # Extract bounding boxes and segmentation masks
    bboxes = result[0]
    segms = result[1]

    # Define palette for visualization
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Visualize all segments
    img_with_segments = visualize_segments(image_path, bboxes, segms, palette)

    # Save or show the result
    if output_path:
        mmcv.imwrite(img_with_segments, output_path)
        print(f"Result saved to: {output_path}")
    else:
        mmcv.imshow(img_with_segments)
        print("Result visualized.")


# Run the testing function
test_image(CONFIG_FILE, CHECKPOINT_FILE, IMAGE_PATH, OUTPUT_PATH)

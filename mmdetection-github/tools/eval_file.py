import argparse

import numpy as np
from mmdet.apis import inference_detector, init_detector

# Paths to your configuration file and pretrained weights
CONFIG_FILE = "configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py"  # Replace with the actual path to your config file
CHECKPOINT_FILE = "epoch_35.pth"  # Replace with the actual path to your .pth file
IMAGE_PATH = "data/eval/image2.jpg"  # Path to the image you want to test
OUTPUT_PATH = "data/eval/output.jpg"  # Path to save the visualization result (optional)


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
    print(f"Testing image: {image_path}")
    model = init_detector(config_file, checkpoint_file, device=device)
    result = inference_detector(model, image_path)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    np.save("bboxes.npy", bboxes)

    model.show_result_part100x(image_path, result, score_thr=0.5, out_file=output_path)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection inference")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--output", default=None, help="Path to save output image (optional)"
    )
    parser.add_argument(
        "--device", default="cuda:0", help='Device to use ("cuda:0" or "cpu")'
    )

    args = parser.parse_args()

    test_image(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        device=args.device,
    )

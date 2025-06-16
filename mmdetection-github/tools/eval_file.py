from mmdet.apis import init_detector, inference_detector
import mmcv

# Paths to your configuration file and pretrained weights
CONFIG_FILE = "configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py"  # Replace with the actual path to your config file
CHECKPOINT_FILE = "epoch_35.pth"  # Replace with the actual path to your .pth file
IMAGE_PATH = "data/eval/image.jpg"  # Path to the image you want to test
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
    # Initialize the model
    model = init_detector(config_file, checkpoint_file, device=device)

    # Perform inference on the image
    result = inference_detector(model, image_path)

    # Visualize and optionally save the result
    print(f"Testing image: {image_path}")
    if output_path:
        model.show_result(image_path, result, out_file=output_path)
        print(f"Result saved to: {output_path}")
    else:
        model.show_result(image_path, result, score_thr=0.5)
        print("Result visualized.")


# Run the testing function
test_image(CONFIG_FILE, CHECKPOINT_FILE, IMAGE_PATH, OUTPUT_PATH)

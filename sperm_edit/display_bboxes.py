import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread("sperm_edit/example.jpg")  # Load image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB


# new_height = 350
# original_height, original_width = image.shape[:2]
# aspect_ratio = original_width / original_height
# new_width = int(new_height * aspect_ratio)
# image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
# Load bounding boxes from .pt file
bboxes = np.load("sperm_edit/bboxes.npy")  # Load tensor

# Print box info for verification
print(f"Loaded {len(bboxes)} bounding boxes")
print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
print(f"First box coordinates: {bboxes[0]}")


# Draw bounding boxes on the image
for box in bboxes:
    # Convert coordinates to integers
    x1, y1, x2, y2, s = box

    print(
        (int(x1), int(y1)),
        (int(x2), int(y2)),
    )
    # Draw rectangle (red color, 2px thickness)
    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color=(255, 0, 0),  # Red in RGB
        thickness=2,
    )


# Display with matplotlib
plt.figure(figsize=(12, 9))
plt.imshow(image)
plt.axis("off")
plt.title(f"Bounding Box Visualization ({len(bboxes)} boxes)")
plt.show()

# Optional: Save result
# cv2.imwrite('output.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

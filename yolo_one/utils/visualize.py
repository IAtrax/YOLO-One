import rerun as rr
import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, labels, scores):
    """
    Draws bounding boxes on an image.

    Args:
        image (np.ndarray): The image to draw on.
        boxes (np.ndarray): The bounding boxes to draw.
        labels (list): The labels for each bounding box.
        scores (list): The scores for each bounding box.
    """
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{label}: {score:.2f}'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def show_image(title, image):
    """
    Shows an image using rerun.

    Args:
        title (str): The title of the image.
        image (np.ndarray): The image to show.
    """
    rr.log_image(title, image)

def log_detections(image, boxes, labels, scores):
    """
    Logs detections to rerun.

    Args:
        image (np.ndarray): The image to log.
        boxes (np.ndarray): The bounding boxes.
        labels (list): The labels for each bounding box.
        scores (list): The scores for each bounding box.
    """
    rr.log("detection", rr.Image(image))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        points = np.array([
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2],  # Bottom-left
            [x1, y1]   # Close the rectangle
        ])
        rr.log(
            f"detection/box_{i}",
            rr.LineStrips2D(points, colors=[(0, 255, 0)] * 5),
            rr.Text(f"{labels[i]}: {scores[i]:.2f}", position=[x1, y1 - 10], color=(0, 255, 0))
        )

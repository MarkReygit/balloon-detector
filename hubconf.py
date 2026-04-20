import torch
import sys
import os

dependencies = ["torch"]

def balloon_detector(pretrained=True, **kwargs):
    """
    YOLOv5s balloon detection model.
    Single class: balloon (class 0).
    Input: BGR frames at 640x640.
    mAP@0.5: 0.902
    """
    yolo_path = os.path.join(os.path.dirname(__file__), "yolov5")
    if os.path.isdir(yolo_path) and yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)

    model_path = os.path.join(os.path.dirname(__file__), "balloon_detector_best.pt")
    model = torch.hub.load(yolo_path, "custom", path=model_path,
                           source="local", verbose=False)
    return model
